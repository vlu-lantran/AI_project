import argparse
import cv2

from anomaly_detection.anomaly_detect import anomaly_detect
from object_detection.Object_Detection import process_video
# from plate_number_detection.plate_number_detect import TrafficCam, get_args
# from plate_number_detection.tracking.deep_sort import DeepSort
# from plate_number_detection.tracking.sort import Sort
# from plate_number_detection.utils.utils import map_label, check_image_size, draw_text, check_legit_plate, \
#     gettime, compute_color, BGR_COLORS, VEHICLES, crop_expanded_plate, correct_plate
# from plate_number_detection.ppocr_onnx import DetAndRecONNXPipeline



if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="""Program for Camera AI detection.
        
    #     Example:
    #     python .\tfs_clone.py --folder_path="$/Professional_Services/Clients/MCA/DebitOrderProcessor/" --mainbranch="trunk" --checkout_folder_name="DebitOrderProcessor"
    #     python .\tfs_clone.py --folder_path="$/Professional_Services/Clients/MCA/ChargePeriodUpdater" --mainbranch="trunk" --checkout_folder_name="ChargePeriodUpdater" --skip_checkout_and_rename
    #     """,
    #     formatter_class=argparse.RawDescriptionHelpFormatter
    # )
    # parser.add_argument("--folder_path", help="The name of the file to process", required=True)
    # parser.add_argument("--mainbranch", help="The name of the main branch in TFS. Case sensitive. Default: trunk", default="trunk")
    # parser.add_argument("--checkout_folder_name", help="Name of the new checkout folder. Default: mainbranch's name", default="")
    # parser.add_argument("--skip_checkout_and_rename", help="Skip checkout all branches and rename", action='store_false')

    # args = parser.parse_args()
    video_path = "input/1.mp4"
    model_path = 'yolov8m.pt'
    output_video_path = 'output_video_4.mp4'
    output_json_path = 'tracking_results_3.json'

    anomaly_detect(video_path, "./", "video")

    process_video(video_path, model_path, output_video_path, output_json_path)
    # opts = get_args()
    # TrafficCam = TrafficCam(opts)
    # TrafficCam.run()