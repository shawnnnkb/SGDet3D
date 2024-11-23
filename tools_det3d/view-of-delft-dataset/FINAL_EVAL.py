from vod.evaluation import Evaluation
import os
from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader
from vod.visualization import Visualization2D
import argparse

def eval_results(test_root, pred_results):
    # When the instance is created, the label locations are required.
    evaluation = Evaluation(test_annotation_file=os.path.join(test_root, 'label_2'))

    # Using the evaluate method, the model can be evaluated on the detection labels.
    results = evaluation.evaluate(result_path=pred_results, current_class=[0, 1, 2])

    print("\nResults: \n"
        f"Entire annotated area | 3d bev aos: \n"
        f"Car: {results['entire_area']['Car_3d_all']:.2f}, {results['entire_area']['Car_bev_all']:.2f}, {results['entire_area']['Car_aos_all']:.2f}\n"
        f"Ped: {results['entire_area']['Pedestrian_3d_all']:.2f}, {results['entire_area']['Pedestrian_bev_all']:.2f}, {results['entire_area']['Pedestrian_aos_all']:.2f} \n"
        f"Cyc: {results['entire_area']['Cyclist_3d_all']:.2f}, {results['entire_area']['Cyclist_bev_all']:.2f}, {results['entire_area']['Cyclist_aos_all']:.2f} \n"
        f"mAP: {((results['entire_area']['Car_3d_all'] + results['entire_area']['Pedestrian_3d_all'] + results['entire_area']['Cyclist_3d_all']) / 3):.2f}, "
            f"{((results['entire_area']['Car_bev_all'] + results['entire_area']['Pedestrian_bev_all'] + results['entire_area']['Cyclist_bev_all']) / 3):.2f}, "
            f"{((results['entire_area']['Car_aos_all'] + results['entire_area']['Pedestrian_aos_all'] + results['entire_area']['Cyclist_aos_all']) / 3):.2f}\n"
        f"Driving corridor area | 3d bev aos: \n"
        f"Car: {results['roi']['Car_3d_all']:.2f}, {results['roi']['Car_bev_all']:.2f}, {results['roi']['Car_aos_all']:.2f}\n"
        f"Ped: {results['roi']['Pedestrian_3d_all']:.2f}, {results['roi']['Pedestrian_bev_all']:.2f}, {results['roi']['Pedestrian_aos_all']:.2f} \n"
        f"Cyc: {results['roi']['Cyclist_3d_all']:.2f}, {results['roi']['Cyclist_bev_all']:.2f}, {results['roi']['Cyclist_aos_all']:.2f} \n"
        f"mAP: {((results['roi']['Car_3d_all'] + results['roi']['Pedestrian_3d_all'] + results['roi']['Cyclist_3d_all']) / 3):.2f}, "
            f"{((results['roi']['Car_bev_all'] + results['roi']['Pedestrian_bev_all'] + results['roi']['Cyclist_bev_all']) / 3):.2f}, "
            f"{((results['roi']['Car_aos_all'] + results['roi']['Pedestrian_aos_all'] + results['roi']['Cyclist_aos_all']) / 3):.2f}, \n"
        f"NOT interested area far distance | 3d bev aos: \n"
        f"Car: {results['not_roi']['Car_3d_all']:.2f}, {results['not_roi']['Car_bev_all']:.2f}, {results['not_roi']['Car_aos_all']:.2f}\n"
        f"Ped: {results['not_roi']['Pedestrian_3d_all']:.2f}, {results['not_roi']['Pedestrian_bev_all']:.2f}, {results['not_roi']['Pedestrian_aos_all']:.2f} \n"
        f"Cyc: {results['not_roi']['Cyclist_3d_all']:.2f}, {results['not_roi']['Cyclist_bev_all']:.2f}, {results['not_roi']['Cyclist_aos_all']:.2f} \n"
        f"mAP: {((results['not_roi']['Car_3d_all'] + results['not_roi']['Pedestrian_3d_all'] + results['not_roi']['Cyclist_3d_all']) / 3):.2f}, "
            f"{((results['not_roi']['Car_bev_all'] + results['not_roi']['Pedestrian_bev_all'] + results['not_roi']['Cyclist_bev_all']) / 3):.2f}, "
            f"{((results['not_roi']['Car_aos_all'] + results['not_roi']['Pedestrian_aos_all'] + results['not_roi']['Cyclist_aos_all']) / 3):.2f}\n"
    )

def Visualization_2D(data_root, output_dir, pred_dir=None):
    output_dir = os.path.join('view-of-delft-dataset', output_dir)
    pred_dir = os.path.join('view-of-delft-dataset', 'pred_dir', pred_dir) if pred_dir!=None else None
    os.makedirs(output_dir, exist_ok=True)
    kitti_locations = KittiLocations(root_dir=data_root, output_dir=output_dir, pred_dir=pred_dir)
    frame_data = FrameDataLoader(kitti_locations=kitti_locations, frame_number="08745")
    vis2d = Visualization2D(frame_data, classes_visualized=['Cyclist', 'Pedestrian', 'Car'])
    vis2d.draw_plot(show_lidar=False,
                    show_radar=True,
                    show_pred=True if pred_dir!=None else False,
                    show_gt=True,
                    min_distance_threshold=1, # 5
                    max_distance_threshold=50, # 20
                    save_figure=True,
                    score_threshold=0.2)

def parse_args():
    parser = argparse.ArgumentParser(description='reprocessing test results using VoD official tools')
    parser.add_argument('--pred_results', default='./tools_det3d/view-of-delft-dataset/pred_results/vod-RadarPillarNet')
    args = parser.parse_args()
    if not 'pillar' in args.pred_results.split('/')[-1].lower():
        args.pred_results = args.pred_results + 'pts_bbox'
    return args
    
if __name__ == '__main__':
    args = parse_args()
    print(args.pred_results)
    eval_results(test_root=os.path.join('./data/VoD', 'radar_5frames', 'testing'), pred_results=args.pred_results)
    # Visualization_2D(data_root, output_dir='visualization_output', pred_dir=None)