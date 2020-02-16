from ImageOptimizer import ImageOptimizer

optimizer = ImageOptimizer(target_image_path=r"images/current_target.png",
                           project_name="ARC9_v2")

optimizer.estimate_target_image()