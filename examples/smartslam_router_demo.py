import argparse
from pathlib import Path

from slam_llm.models.smartslam_router import SmartSLAMRouter


def main():
    parser = argparse.ArgumentParser(
        description="SmartSLAM Adaptive Multimodal Routing Demo"
    )
    parser.add_argument(
        "audio",
        help="Path to an audio file"
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    router = SmartSLAMRouter()
    result = router.route(audio_path)

    print()
    print("=" * 60)
    print("SmartSLAM Adaptive Multimodal Routing")
    print("=" * 60)
    print(f"Input audio       : {audio_path}")
    print(f"Detected modality : {result['modality']}")
    print(f"Selected encoder  : {result['encoder']}")
    print(f"Routing confidence: {result['confidence']:.2f}")

    print()
    print("Acoustic features:")
    for name, value in result["features"].items():
        if name == "duration_seconds":
            print(f"  {name:24s}: {value:.3f}")
        elif name == "spectral_centroid_hz":
            print(f"  {name:24s}: {value:.2f}")
        else:
            print(f"  {name:24s}: {value:.6f}")

    print()
    print("Selected SLAM-LLM profile:")
    for name, value in result["slam_llm_profile"].items():
        print(f"  {name:28s}: {value}")

    print()
    print("Routing reasoning:")
    for reason in result["reasoning"]:
        print(f"  - {reason}")

    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
