# InfiniteTalk Nodes for Griptape

This library provides Griptape Nodes for [InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk), an audio-driven video generation model. Generate talking videos from a still image and driving audio, or dub an existing video with new audio while keeping the speaker's lips in sync, all within your Griptape workflows.

The models run locally on your own hardware and require a CUDA-capable GPU.

## Features

- **Image-to-Video**: Animate a still image of a person so they appear to speak your driving audio, with synchronized lip movement and expressions.
- **Video-to-Video (dubbing)**: Take an existing video and new audio, and re-sync the speaker's lips to the new audio.
- **Clip and streaming modes**: Generate a single segment (`clip`) or longer videos (`streaming`).
- **Selectable resolution**: Choose the Wan-AI base model to target 480p or 720p output.
- **Local execution**: All model weights are downloaded from Hugging Face and run on your own GPU.

## Requirements

- A CUDA-capable NVIDIA GPU.
- The following models are downloaded automatically from Hugging Face on first use:
  - **Base model**: `Wan-AI/Wan2.1-I2V-14B-480P` or `Wan-AI/Wan2.1-I2V-14B-720P` (determines output resolution)
  - **Audio encoder**: `TencentGameMate/chinese-wav2vec2-base`
  - **InfiniteTalk weights**: `MeiGen-AI/InfiniteTalk`

## Installation

1. Clone this repository into your Griptape Nodes workspace directory:

```bash
# Navigate to your workspace directory
# On Mac or Linux you can use the command below to print your workspace directory
cd $(gtn config show | grep workspace_directory | cut -d'"' -f4)
# On Windows, the default workspace directory is a directory named GriptapeNodes in your home directory.
# Usually this is C:\Users\<username>\GriptapeNodes

# Clone the repository (with submodules)
git clone --recurse-submodules https://github.com/griptape-ai/griptape-nodes-infinite-talk-library.git
```

2. Install dependencies:

```bash
cd griptape-nodes-infinite-talk-library
uv sync
```

## Add your library to your installed Engine!

If you haven't already installed your Griptape Nodes engine, follow the installation steps [HERE](https://github.com/griptape-ai/griptape-nodes).
After you've completed those and you have your engine up and running:

1. Copy the path to your `griptape_nodes_library.json` file within the `griptape_nodes_infinite_talk_library` directory. Right click on the file, and `Copy Path` (Not `Copy Relative Path`).
2. Start up the engine!
3. Navigate to settings.
4. Open your settings and go to the App Events tab. Add an item in **Libraries to Register**.
5. Paste your copied `griptape_nodes_library.json` path from earlier into the new item.
6. Exit out of Settings. It will save automatically!
7. Open up the **Libraries** dropdown on the left sidebar.
8. Your newly registered library should appear! Drag and drop nodes to use them!

## Available Nodes

### InfiniteTalk Image to Video

Generate a talking video from a reference image and driving audio. The person in the image appears to speak the audio with synchronized lip movement.

- **Image**: Reference image of the person to animate.
- **Audio**: Driving audio for lip sync and expressions.
- **Prompt**: Text description of the video content.
- **Mode**: Generation mode, `clip` for a single segment or `streaming` for longer videos.
- **Base Model**: Wan-AI model for video generation (determines resolution: 480p or 720p).
- **Audio Encoder**: Wav2Vec model for audio encoding.
- **InfiniteTalk Weights**: InfiniteTalk model weights.

Outputs the generated talking **Video**.

### InfiniteTalk Video to Video

Dub an existing video with new audio, re-synchronizing the speaker's lips to the new driving audio.

- **Input Video**: Reference video to dub with new audio.
- **Audio**: New driving audio for lip sync.
- **Prompt**: Text description of the video content.
- **Mode**: Generation mode, `clip` or `streaming`.
- **Base Model**: Wan-AI model for video generation (determines resolution).
- **Audio Encoder**: Wav2Vec model for audio encoding.
- **InfiniteTalk Weights**: InfiniteTalk model weights.

Outputs the generated dubbed **Video**.

## Example Workflows

### Animate a Portrait

1. Add an **InfiniteTalk Image to Video** node.
2. Connect a reference image of the person to the **Image** input.
3. Connect your driving audio to the **Audio** input.
4. Optionally set a **Prompt** describing the scene, and choose a **Mode** (`clip` for short clips, `streaming` for longer output).
5. Select the **Base Model** for your desired resolution.
6. Run the workflow. The talking video is available on the **Video** output.

### Dub an Existing Video

1. Add an **InfiniteTalk Video to Video** node.
2. Connect the video you want to dub to the **Input Video** input.
3. Connect the new audio to the **Audio** input.
4. Run the workflow. The dubbed video, with lips re-synced to the new audio, is available on the **Video** output.

## Troubleshooting

**"CUDA not available"**
- InfiniteTalk requires a CUDA-capable NVIDIA GPU. Confirm your drivers and CUDA runtime are installed.

**Missing models**
- The node validates that all required models are downloaded before running. If a model is missing, download it via the model dropdowns before running the node.

**Slow first run**
- Model weights are downloaded from Hugging Face on first use and cached locally. Subsequent runs are faster.

## License

The Griptape Nodes integration code in this repository is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details. The underlying [InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk) model and the Wan-AI models are subject to their own license terms.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
