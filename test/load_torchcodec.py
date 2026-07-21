from torchcodec.decoders import VideoDecoder


decoder = VideoDecoder("data/lerobot/AI2_alphabot_2_arrange_teaset_test/videos/chunk-000/observation.images.cam_chest_depth/episode_000000.mp4", device="cuda", seek_mode="approximate", output_dtype="auto")
meta = decoder.metadata

print(meta.pixel_format)

frame = decoder.get_frame_at(0).data[0, ::4, ::4] * 4096

import matplotlib.pyplot as plt
plt.imshow(frame.cpu().numpy(), cmap="gray")
plt.show()
