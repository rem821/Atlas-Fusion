#pragma once

#define Lidar_Laser_Approx_And_Seg context_.getFunctionalityFlags().lidar_laser_approximations_and_segmentation_
#define RGB_Detection_To_IR_Projection context_.getFunctionalityFlags().rgb_to_ir_detection_projection_
#define Depth_Map_For_IR context_.getFunctionalityFlags().generate_depth_map_for_ir_
#define Depth_Map_For_RGB_Left_Front context_.getFunctionalityFlags().generate_depth_map_for_rgb_left_front_
#define Depth_Map_For_RGB_Virtual context_.getFunctionalityFlags().generate_depth_map_for_rgb_virtual_
#define Short_Term_Lidar_Aggregation context_.getFunctionalityFlags().short_term_lidar_aggregation_
#define Lidar_Colorization context_.getFunctionalityFlags().colorized_lidar_
#define Global_Lidar_Aggregation context_.getFunctionalityFlags().global_lidar_aggregation_
#define Center_Lidar_Only context_.getFunctionalityFlags().use_only_central_lidar_