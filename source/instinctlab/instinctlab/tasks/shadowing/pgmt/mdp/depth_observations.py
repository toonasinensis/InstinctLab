"""Single current camera image; history/reset timing belongs to the native sensor."""

def depth_image(env, sensor_name="camera"):
    image = env.scene[sensor_name].data.output["distance_to_image_plane_noised"].torch
    return image.permute(0, 3, 1, 2).clone()
