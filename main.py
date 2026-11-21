import signal
import sys
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib


def build_sender_pipeline(
    device: str = "/dev/video4",
    host: str = "127.0.0.1",
    port: int = 5005,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    bitrate_kbps: int = 2500,
) -> Gst.Pipeline:
    """
    Captures MJPEG from /dev/video4 at a verified supported mode, decodes it,
    encodes H.264 with zero-latency tuning, and streams over RTP/UDP.
    """
    pipeline_str = (
        f"v4l2src device={device} do-timestamp=true ! "
        f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
        f"jpegdec ! "
        f"videoconvert ! "
        f"video/x-raw,format=I420 ! "
        f"x264enc tune=zerolatency speed-preset=ultrafast bitrate={bitrate_kbps} key-int-max={fps} ! "
        f"rtph264pay config-interval=1 pt=96 ! "
        f"udpsink host={host} port={port} sync=false async=false"
    )
    return Gst.parse_launch(pipeline_str)


def build_receiver_pipeline(
    port: int = 5005,
    jitter_latency_ms: int = 20,
) -> Gst.Pipeline:
    """
    Receives RTP/H.264 over UDP, keeps jitter buffer minimal, decodes, and displays.
    """
    pipeline_str = (
        f"udpsrc port={port} caps=\"application/x-rtp,media=video,encoding-name=H264,payload=96\" ! "
        f"rtpjitterbuffer latency={jitter_latency_ms} drop-on-latency=true ! "
        f"rtph264depay ! "
        f"avdec_h264 ! "
        f"videoconvert ! "
        f"autovideosink sync=false"
    )
    return Gst.parse_launch(pipeline_str)


def attach_bus_handlers(pipeline: Gst.Pipeline, name: str, loop: GLib.MainLoop) -> None:
    """Listens for GStreamer errors and end-of-stream events."""
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_message(_bus: Gst.Bus, msg: Gst.Message) -> None:
        msg_type = msg.type
        if msg_type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            print(f"[{name}] GStreamer Error: {err.message}", file=sys.stderr)
            if debug:
                print(f"[{name}] Debug: {debug}", file=sys.stderr)
            loop.quit()
        elif msg_type == Gst.MessageType.EOS:
            print(f"[{name}] End of stream (EOS).")
            loop.quit()

    bus.connect("message", on_message)


def main() -> None:
    Gst.init(None)
    loop = GLib.MainLoop()

    sender = build_sender_pipeline(
        device="/dev/video4",
        host="127.0.0.1",
        port=5005,
        width=1280,
        height=720,
        fps=30,
        bitrate_kbps=2500,
    )
    receiver = build_receiver_pipeline(
        port=5005,
        jitter_latency_ms=20,
    )

    attach_bus_handlers(sender, "Sender", loop)
    attach_bus_handlers(receiver, "Receiver", loop)

    def on_sigint(*_args) -> None:
        print("\nStopping (Ctrl+C)...")
        loop.quit()

    signal.signal(signal.SIGINT, on_sigint)

    print("Starting GStreamer pipelines on /dev/video4 (MJPG 1280x720@30) -> udp://127.0.0.1:5005 ...")
    ret_rx = receiver.set_state(Gst.State.PLAYING)
    ret_tx = sender.set_state(Gst.State.PLAYING)

    if ret_rx == Gst.StateChangeReturn.FAILURE or ret_tx == Gst.StateChangeReturn.FAILURE:
        print("Failed to set one of the pipelines to PLAYING.", file=sys.stderr)
        sender.set_state(Gst.State.NULL)
        receiver.set_state(Gst.State.NULL)
        sys.exit(1)

    try:
        loop.run()
    finally:
        print("Cleaning up pipelines...")
        sender.set_state(Gst.State.NULL)
        receiver.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()