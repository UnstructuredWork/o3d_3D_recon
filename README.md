# o3d_3D_recon
Open3D Realtime 3D reconstruction based on AI Model

## panoptic_model
### 1) [download](https://drive.google.com/file/d/1X9vFCsLzAAvEHxAQA2yaZBHKyjpYz84p/view?usp=drive_link)
or NAS/Dataset/panotpic_model/mask2former_r50_8xb2-lsj-50e_coco-panoptic.zip  
After unzip the ~-panotic.zip file, move to <root>/o3d_3D_recon/panoptic/mask2former_custom/

### 2) classes
#### thing_classes
##### 1. background 2. circular_valve 3. straight_valve 4. fire_hydrant 5. fire_extinguisher 
##### 6. fire_extinguisher_box 7. manometer 8. control_box 9. emergency_stop_switch 10. warning_light
#### stuff_classes
##### 11. wall-other-merged 12. floor-other-merged

***
## File Path
create a 'mask2former_custom' directory and put download files in it
```bash
o3d_3D_recon
├── o3d_recon
├── panoptic
│ ├── _base_
│ ├── mask2former
│ ├── mask2former_custom
│ │ ├── mask2former_r50_8xb2-lsj-50e_coco-panoptic.py
│ │ └── iter_5000.pth
│ └── panoptic.py 
├── test
├── README.md
├── gui.py
├── install.sh
├── main.py
├── requirements.txt
├── ros_test.py
└── setup.py
```
***
