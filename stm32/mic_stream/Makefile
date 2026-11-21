# Build and flash mic_stream for the B-L475E-IOT01A.
# Needs: arm-none-eabi-gcc + newlib (Debian: gcc-arm-none-eabi libnewlib-arm-none-eabi) and openocd.
#   make            build build/mic_stream.elf
#   make flash      program the board through its ST-LINK (USB)
#   make restore    put back the original firmware from BACKUP

CC      = arm-none-eabi-gcc
SIZE    = arm-none-eabi-size
OBJCOPY = arm-none-eabi-objcopy
OPENOCD = openocd -f board/st_b-l475e-iot01a.cfg
BACKUP ?= $(HOME)/stm32_backup/original_firmware_bank0.bin

ARCH    = -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard
CFLAGS  = $(ARCH) -O2 -g3 -std=c11 -Wall -Wextra -ffunction-sections -fdata-sections \
          -DSTM32L475xx -Isrc -Ithird_party/cmsis
LDFLAGS = $(ARCH) -T stm32l475vg.ld -Wl,--gc-sections -Wl,-Map=build/mic_stream.map \
          -specs=nano.specs -specs=nosys.specs

SRCS = src/main.c third_party/cmsis/system_stm32l4xx.c
ASMS = third_party/cmsis/startup_stm32l475xx.s
OBJS = $(SRCS:%.c=build/%.o) $(ASMS:%.s=build/%.o)

all: build/mic_stream.elf build/mic_stream.bin

build/%.o: %.c src/fir_coeffs.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

build/%.o: %.s
	@mkdir -p $(dir $@)
	$(CC) $(ARCH) -c $< -o $@

build/mic_stream.elf: $(OBJS) stm32l475vg.ld
	$(CC) $(LDFLAGS) $(OBJS) -o $@
	$(SIZE) $@

build/mic_stream.bin: build/mic_stream.elf
	$(OBJCOPY) -O binary $< $@

flash: build/mic_stream.elf
	$(OPENOCD) -c "program build/mic_stream.elf verify reset exit"

restore:
	$(OPENOCD) -c "program $(BACKUP) 0x08000000 verify reset exit"

clean:
	rm -rf build

.PHONY: all flash restore clean
