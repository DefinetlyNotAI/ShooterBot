# Detection IDs

The detector stack uses the standard COCO class order for supported YOLO object classes. These IDs match `src/coco.py`.

|    ID | Class         | ID | Class          | ID | Class        |
| ----: |---------------| -: | -------------- | -: | ------------ |
| ~~0~~ | ~~person~~*   | 27 | tie            | 54 | donut        |
|     1 | bicycle       | 28 | suitcase       | 55 | cake         |
|     2 | car           | 29 | frisbee        | 56 | chair        |
|     3 | motorcycle    | 30 | skis           | 57 | couch        |
|     4 | airplane      | 31 | snowboard      | 58 | potted plant |
|     5 | bus           | 32 | sports ball    | 59 | bed          |
|     6 | train         | 33 | kite           | 60 | dining table |
|     7 | truck         | 34 | baseball bat   | 61 | toilet       |
|     8 | boat          | 35 | baseball glove | 62 | tv           |
|     9 | traffic light | 36 | skateboard     | 63 | laptop       |
|    10 | fire hydrant  | 37 | surfboard      | 64 | mouse        |
|    11 | stop sign     | 38 | tennis racket  | 65 | remote       |
|    12 | parking meter | 39 | bottle         | 66 | keyboard     |
|    13 | bench         | 40 | wine glass     | 67 | cell phone   |
|    14 | bird          | 41 | cup            | 68 | microwave    |
|    15 | cat           | 42 | fork           | 69 | oven         |
|    16 | dog           | 43 | knife          | 70 | toaster      |
|    17 | horse         | 44 | spoon          | 71 | sink         |
|    18 | sheep         | 45 | bowl           | 72 | refrigerator |
|    19 | cow           | 46 | banana         | 73 | book         |
|    20 | elephant      | 47 | apple          | 74 | clock        |
|    21 | bear          | 48 | sandwich       | 75 | vase         |
|    22 | zebra         | 49 | orange         | 76 | scissors     |
|    23 | giraffe       | 50 | broccoli       | 77 | teddy bear   |
|    24 | backpack      | 51 | carrot         | 78 | hair drier   |
|    25 | umbrella      | 52 | hot dog        | 79 | toothbrush   |
|    26 | handbag       | 53 | pizza          |    |              |

*Note: The `person` class is disabled by default in the detector stack. To enable it, set `DETECTOR_PERSON_ENABLED` to `True` in your configuration file. Reason is "face" detection is better.