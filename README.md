# 7SMetrics

Calculates the top seven metrics used to validate targeting results.

## Requirements
  * OpenCV-python == 4.1.1.26
  * Numba == 0.45.1
 
## The Seven Segmentation Metrics
 * Accuracy score
 * Precision
 * Dice Coefficient
 * Jaccard Index
 * Matthew Correlation Coefficient
 * Sensitivity coefficient
 * Specificity coefficient
 
 
 ### Usage:
 This tool needs the directory of two main folders. One of the folders will contain the results of the proposed segmentation. The other folder will contain the ground truth provided by the dataset used.
 

 To execute 'main.py', it is necessary, via command-line, to enter the directory of the two folders as arguments for execution. As shown below: 

$ python main.py -seg directory_segmentation -GT directory_ground_truth
 
 
 ### Output
 The 'imgs.txt' file contains the values ​​of each metric, separated by segmented image. The 'general_result.txt' file contains the average values ​​of each metric with their standard deviations for the entire set of images. 
 
