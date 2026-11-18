# vision_inspect

`robot_runtime_v2`'s `vision.inspect`: one current or saved camera image, in RGB or a depth colour map, whole or cropped, as a
512 px thumbnail or at full size. The result binds the image to its source hashes. A depth colour map is a picture, not a
measurement: its statistics are in `depth_stats`, and `geometry.query` measures.
