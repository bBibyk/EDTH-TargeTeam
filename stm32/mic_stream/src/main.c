/*
 * mic_stream: B-L475E-IOT01A MEMS microphone -> raw 32 kHz PCM over the ST-LINK USB serial port.
 *
 * The board only digitises and streams sound. The Raspberry Pi does the mel
 * spectrogram and the model (demo/raspberry/audio_wake), exactly like training.
 *
 *   MP34DT01 PDM mic: PE9 = DFSDM1_CKOUT (1.6 MHz), PE7 = DFSDM1_DATIN2, channel 2, falling edge
 *                     (same wiring and clock as the team's original firmware)
 *   DFSDM1 filter 0:  sinc5, FOSR 25, fast continuous mode -> 64 kHz, DMA1 ch4 -> RAM (circular)
 *   CPU (M4F):        64-tap FIR (low-pass 14 kHz + sinc5 droop compensation), keep every 2nd
 *                     sample -> 32 kHz, DC-blocking high-pass (~10 Hz), +12 dB, round to int16
 *   USART1:           PB6 TX / PB7 RX, 1 Mbaud, DMA2 ch6 -> ST-LINK VCP -> /dev/ttyACM0 on the Pi
 *
 * Packet, little-endian, one every 8 ms:
 *   A5 5A | seq u16 | n u16 (=256) | n x int16 samples | checksum u16
 *   checksum = seq + n + all samples (each as u16), mod 2^16
 * `seq` goes up by one per packet (also for packets dropped on the board), so the Pi sees gaps.
 * At boot, before the packets, one text line: "$MIC,32000,1,S16LE,<version>\n".
 */
#include <stdint.h>
#include <string.h>

#include "stm32l4xx.h"
#include "fir_coeffs.h"

#define FW_VERSION    "mic_stream-2"
#define SYSCLK_HZ     80000000u
#define BAUD          1000000u
#define CKOUT_DIV     50u                 /* 80 MHz / 50 = 1.6 MHz PDM clock          */
#define DFSDM_FOSR    25u                 /* 1.6 MHz / 25 = 64 kHz                    */
#define DFSDM_ORDER   5u                  /* sinc5                                    */
#define DFSDM_RBS     1u                  /* 25^5 = 9.77e6 > 2^23 -> shift right by 1 */
#define FULL_SCALE    (9765625.0f / 2.0f) /* DFSDM output for an all-ones PDM stream  */
#define GAIN          4.0f                /* +12 dB: int16 clips at about 108 dB SPL  */
#define HP_POLE       0.998f              /* DC blocker, about 10 Hz at 32 kHz        */
#define OUT_N         256u                /* samples per packet = 8 ms at 32 kHz      */
#define IN_N          (2u * OUT_N)        /* 64 kHz samples per DMA half-buffer       */
#define PKT_LEN       (6u + 2u * OUT_N + 2u)

static volatile int32_t dfsdm_buf[2u * IN_N];          /* DMA target, two halves     */
static float fir_line[FIR_NTAPS - 1u + IN_N];           /* FIR history + new block    */
static uint8_t pkt[2][PKT_LEN] __attribute__((aligned(4)));
static unsigned pkt_cur;
static uint16_t seq;
static float hp_x1, hp_y1;
volatile uint32_t dropped_packets;                      /* readable with a debugger   */

static void clock_init(void)
{
    RCC->APB1ENR1 |= RCC_APB1ENR1_PWREN;
    (void)RCC->APB1ENR1;
    FLASH->ACR = FLASH_ACR_LATENCY_4WS | FLASH_ACR_PRFTEN | FLASH_ACR_ICEN | FLASH_ACR_DCEN;
    while ((FLASH->ACR & FLASH_ACR_LATENCY) != FLASH_ACR_LATENCY_4WS) {
    }

    /* 32.768 kHz crystal -> MSI hardware auto-calibration, so 32 kHz really is 32 kHz. */
    PWR->CR1 |= PWR_CR1_DBP;
    RCC->BDCR |= RCC_BDCR_LSEON;
    for (uint32_t t = 0; t < 4000000u && !(RCC->BDCR & RCC_BDCR_LSERDY); t++) {
    }
    if (RCC->BDCR & RCC_BDCR_LSERDY) {
        RCC->CR |= RCC_CR_MSIPLLEN;
    }

    /* MSI 4 MHz -> PLL (M=1, N=40, R=2) -> 80 MHz SYSCLK. */
    RCC->PLLCFGR = RCC_PLLCFGR_PLLSRC_MSI | (40u << RCC_PLLCFGR_PLLN_Pos) | RCC_PLLCFGR_PLLREN;
    RCC->CR |= RCC_CR_PLLON;
    while (!(RCC->CR & RCC_CR_PLLRDY)) {
    }
    RCC->CFGR = (RCC->CFGR & ~RCC_CFGR_SW) | RCC_CFGR_SW_PLL;
    while ((RCC->CFGR & RCC_CFGR_SWS) != RCC_CFGR_SWS_PLL) {
    }
    SystemCoreClock = SYSCLK_HZ;
}

static void gpio_af(GPIO_TypeDef *g, unsigned pin, unsigned af, unsigned speed)
{
    g->MODER = (g->MODER & ~(3u << (2u * pin))) | (2u << (2u * pin));
    g->OSPEEDR = (g->OSPEEDR & ~(3u << (2u * pin))) | (speed << (2u * pin));
    unsigned shift = (pin & 7u) * 4u;
    g->AFR[pin >> 3] = (g->AFR[pin >> 3] & ~(0xFu << shift)) | (af << shift);
}

static void uart_init(void)
{
    RCC->APB2ENR |= RCC_APB2ENR_USART1EN;
    gpio_af(GPIOB, 6, 7, 2);
    gpio_af(GPIOB, 7, 7, 2);
    USART1->CR1 = 0;
    USART1->BRR = SYSCLK_HZ / BAUD;
    USART1->CR3 = USART_CR3_DMAT;
    USART1->CR1 = USART_CR1_TE | USART_CR1_UE;

    DMA2_CSELR->CSELR = (DMA2_CSELR->CSELR & ~DMA_CSELR_C6S) | (2u << DMA_CSELR_C6S_Pos); /* USART1_TX */
    DMA2_Channel6->CCR = 0;
    DMA2_Channel6->CPAR = (uint32_t)&USART1->TDR;
    DMA2_Channel6->CCR = DMA_CCR_MINC | DMA_CCR_DIR | DMA_CCR_PL_0; /* 8-bit, memory -> USART */
}

static void uart_puts_blocking(const char *s)
{
    while (*s) {
        while (!(USART1->ISR & USART_ISR_TXE)) {
        }
        USART1->TDR = (uint8_t)*s++;
    }
    while (!(USART1->ISR & USART_ISR_TC)) {
    }
}

static void uart_send_dma(const uint8_t *data, uint32_t len)
{
    if ((DMA2_Channel6->CCR & DMA_CCR_EN) && DMA2_Channel6->CNDTR != 0u) {
        dropped_packets++; /* previous packet still going out: USART too slow */
        return;
    }
    DMA2_Channel6->CCR &= ~DMA_CCR_EN;
    DMA2->IFCR = DMA_IFCR_CGIF6;
    DMA2_Channel6->CMAR = (uint32_t)data;
    DMA2_Channel6->CNDTR = len;
    DMA2_Channel6->CCR |= DMA_CCR_EN;
}

static void mic_init(void)
{
    RCC->APB2ENR |= RCC_APB2ENR_DFSDM1EN;
    RCC->CCIPR &= ~RCC_CCIPR_DFSDM1SEL;           /* DFSDM kernel clock = PCLK2 = 80 MHz */
    gpio_af(GPIOE, 9, 6, 3);                      /* DFSDM1_CKOUT  */
    gpio_af(GPIOE, 7, 6, 3);                      /* DFSDM1_DATIN2 */

    /* Channel 2: SPI input, clock from the internal CKOUT, sampled on the falling edge. */
    DFSDM1_Channel2->CHCFGR2 = DFSDM_RBS << DFSDM_CHCFGR2_DTRBS_Pos;
    DFSDM1_Channel2->CHCFGR1 = (1u << DFSDM_CHCFGR1_SPICKSEL_Pos) | (1u << DFSDM_CHCFGR1_SITP_Pos) |
                               DFSDM_CHCFGR1_CHEN;
    /* Global settings live in channel 0: CKOUT divider, then enable the interface. */
    DFSDM1_Channel0->CHCFGR1 = (CKOUT_DIV - 1u) << DFSDM_CHCFGR1_CKOUTDIV_Pos;
    DFSDM1_Channel0->CHCFGR1 |= DFSDM_CHCFGR1_DFSDMEN;

    /* DFSDM regular data -> DMA1 channel 4 (request 0), circular, two halves. */
    DMA1_CSELR->CSELR &= ~DMA_CSELR_C4S;
    DMA1_Channel4->CCR = 0;
    DMA1_Channel4->CPAR = (uint32_t)&DFSDM1_Filter0->FLTRDATAR;
    DMA1_Channel4->CMAR = (uint32_t)dfsdm_buf;
    DMA1_Channel4->CNDTR = 2u * IN_N;
    DMA1_Channel4->CCR = DMA_CCR_PL_1 | DMA_CCR_MSIZE_1 | DMA_CCR_PSIZE_1 | DMA_CCR_MINC |
                         DMA_CCR_CIRC | DMA_CCR_HTIE | DMA_CCR_TCIE | DMA_CCR_TEIE;
    DMA1_Channel4->CCR |= DMA_CCR_EN;
    NVIC_SetPriority(DMA1_Channel4_IRQn, 1);
    NVIC_EnableIRQ(DMA1_Channel4_IRQn);

    /* Filter 0: channel 2, sinc5, FOSR 25, IOSR 1, fast continuous conversion with DMA. */
    DFSDM1_Filter0->FLTCR1 = 0;
    DFSDM1_Filter0->FLTFCR = (DFSDM_ORDER << DFSDM_FLTFCR_FORD_Pos) | ((DFSDM_FOSR - 1u) << DFSDM_FLTFCR_FOSR_Pos);
    DFSDM1_Filter0->FLTCR1 = (2u << DFSDM_FLTCR1_RCH_Pos) | DFSDM_FLTCR1_RCONT | DFSDM_FLTCR1_RDMAEN |
                             DFSDM_FLTCR1_FAST;
    DFSDM1_Filter0->FLTCR1 |= DFSDM_FLTCR1_DFEN;
    DFSDM1_Filter0->FLTCR1 |= DFSDM_FLTCR1_RSWSTART;
}

/* 512 samples at 64 kHz -> FIR + decimate by 2 -> 256 int16 at 32 kHz -> one packet. */
static void process_block(const volatile int32_t *src)
{
    float *line_new = &fir_line[FIR_NTAPS - 1u];
    for (unsigned i = 0; i < IN_N; i++) {
        line_new[i] = (float)(src[i] >> 8) * (1.0f / FULL_SCALE); /* data is in bits 31..8 */
    }

    uint8_t *p = pkt[pkt_cur];
    int16_t *out = (int16_t *)(p + 6);
    uint16_t sum = (uint16_t)(seq + OUT_N); /* the header is covered too */
    for (unsigned k = 0; k < OUT_N; k++) {
        const float *x = &fir_line[2u * k + 1u];
        float acc = 0.0f;
        for (unsigned j = 0; j < FIR_NTAPS; j++) {
            acc += fir_coeffs[j] * x[j]; /* symmetric taps: no need to reverse */
        }
        float y = acc - hp_x1 + HP_POLE * hp_y1;
        hp_x1 = acc;
        hp_y1 = y;
        float s = y * (GAIN * 32767.0f);
        s = s > 32767.0f ? 32767.0f : (s < -32768.0f ? -32768.0f : s);
        int16_t v = (int16_t)(s >= 0.0f ? s + 0.5f : s - 0.5f);
        out[k] = v;
        sum = (uint16_t)(sum + (uint16_t)v);
    }
    memmove(fir_line, &fir_line[IN_N], (FIR_NTAPS - 1u) * sizeof(float));

    p[0] = 0xA5;
    p[1] = 0x5A;
    p[2] = (uint8_t)seq;
    p[3] = (uint8_t)(seq >> 8);
    p[4] = (uint8_t)OUT_N;
    p[5] = (uint8_t)(OUT_N >> 8);
    p[PKT_LEN - 2u] = (uint8_t)sum;
    p[PKT_LEN - 1u] = (uint8_t)(sum >> 8);
    seq++;
    uart_send_dma(p, PKT_LEN);
    pkt_cur ^= 1u;
}

void DMA1_Channel4_IRQHandler(void)
{
    uint32_t isr = DMA1->ISR;
    if (isr & DMA_ISR_TEIF4) {
        DMA1->IFCR = DMA_IFCR_CTEIF4;
    }
    if (isr & DMA_ISR_HTIF4) {
        DMA1->IFCR = DMA_IFCR_CHTIF4;
        process_block(&dfsdm_buf[0]);
    }
    if (isr & DMA_ISR_TCIF4) {
        DMA1->IFCR = DMA_IFCR_CTCIF4;
        process_block(&dfsdm_buf[IN_N]);
    }
}

int main(void)
{
    clock_init();
    RCC->AHB2ENR |= RCC_AHB2ENR_GPIOBEN | RCC_AHB2ENR_GPIOEEN;
    RCC->AHB1ENR |= RCC_AHB1ENR_DMA1EN | RCC_AHB1ENR_DMA2EN;
    (void)RCC->AHB1ENR;

    uart_init();
    uart_puts_blocking("\n$MIC,32000,1,S16LE," FW_VERSION "\n");
    mic_init();

    for (;;) {
        __WFI(); /* everything happens in the DMA interrupt */
    }
}
