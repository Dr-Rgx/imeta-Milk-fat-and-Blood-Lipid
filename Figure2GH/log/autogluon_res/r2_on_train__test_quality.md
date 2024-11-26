/home/luqy21/anaconda3/envs/lqtorch/bin/python3.8 /home/luqy21/.pycharm_helpers/pydev/pydevconsole.py --mode=client
--host=localhost --port=34221
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/home/luqy21/source/medical-data-analysis/'])
PyDev console: starting.
Python 3.8.13 (default, Mar 28 2022, 11:38:47)
[GCC 7.5.0] on linux
runfile('/home/luqy21/source/medical-data-analysis/Auto_ML_for_reg.py',
wdir='/home/luqy21/source/medical-data-analysis/')
IPython could not be loaded!
USE_MY_DATA = True
-------------------- 0 ------------------------
load y_index:0, y:CHO
0 2.59
1 2.59
2 2.73
3 1.39
4 2.46
...
90 3.49
91 3.47
92 3.79
93 4.11
94 4.25
Name: CHO, Length: 95, dtype: float64
model score_test score_val pred_time_test pred_time_val fit_time pred_time_test_marginal pred_time_val_marginal
fit_time_marginal stack_level can_infer fit_order
0 WeightedEnsemble_L2 0.561847 0.509174 45.452639 14.485278 375.359302 0.006283 0.000355 0.165315 2 True 8
1 ExtraTreesMSE_BAG_L1 0.560193 0.508961 0.320801 0.111629 1.954022 0.320801 0.111629 1.954022 1 True 6
2 RandomForestMSE_BAG_L1 0.559534 0.472931 0.242021 0.115026 1.968419 0.242021 0.115026 1.968419 1 True 5
3 LightGBMLarge_BAG_L1 0.492097 0.430550 25.592313 7.690271 236.076543 25.592313 7.690271 236.076543 1 True 7
4 LightGBM_BAG_L1 0.388966 0.434058 19.011647 6.867910 130.991790 19.011647 6.867910 130.991790 1 True 4
5 LightGBMXT_BAG_L1 0.381251 0.446566 19.533241 6.683022 137.163421 19.533241 6.683022 137.163421 1 True 3
6 KNeighborsDist_BAG_L1 0.057023 0.163724 0.234193 0.109837 0.518126 0.234193 0.109837 0.518126 1 True 2
7 KNeighborsUnif_BAG_L1 0.052808 0.153654 0.256649 0.108896 0.513985 0.256649 0.108896 0.513985 1 True 1
model score_val pred_time_val fit_time pred_time_val_marginal fit_time_marginal stack_level can_infer fit_order
0 WeightedEnsemble_L2 0.509174 14.485278 375.359302 0.000355 0.165315 2 True 8
1 ExtraTreesMSE_BAG_L1 0.508961 0.111629 1.954022 0.111629 1.954022 1 True 6
2 RandomForestMSE_BAG_L1 0.472931 0.115026 1.968419 0.115026 1.968419 1 True 5
3 LightGBMXT_BAG_L1 0.446566 6.683022 137.163421 6.683022 137.163421 1 True 3
4 LightGBM_BAG_L1 0.434058 6.867910 130.991790 6.867910 130.991790 1 True 4
5 LightGBMLarge_BAG_L1 0.430550 7.690271 236.076543 7.690271 236.076543 1 True 7
6 KNeighborsDist_BAG_L1 0.163724 0.109837 0.518126 0.109837 0.518126 1 True 2
7 KNeighborsUnif_BAG_L1 0.153654 0.108896 0.513985 0.108896 0.513985 1 True 1
on test {'r2': 0.5618473181789708, 'root_mean_squared_error': -0.665720505339987, 'mean_squared_error':
-0.4431837912301276, 'mean_absolute_error': -0.5445919980500873, 'pearsonr': 0.7690717602934865, '
median_absolute_error': -0.4433804321289063}
-------------------- 1 ------------------------
load y_index:1, y:TG
0 1.80
1 1.25
2 2.07
3 0.95
4 1.33
...
90 1.85
91 1.30
92 1.25
93 1.66
94 1.76
Name: TG, Length: 95, dtype: float64
model score_test score_val pred_time_test pred_time_val fit_time pred_time_test_marginal pred_time_val_marginal
fit_time_marginal stack_level can_infer fit_order
0 LightGBMXT_BAG_L1 -0.141338 0.184157 20.916243 5.652621 110.592622 20.916243 5.652621 110.592622 1 True 3
1 ExtraTreesMSE_BAG_L1 -0.149479 0.159837 0.349716 0.108956 2.480520 0.349716 0.108956 2.480520 1 True 6
2 LightGBM_BAG_L1 -0.166579 0.223466 16.613479 5.985353 113.911626 16.613479 5.985353 113.911626 1 True 4
3 WeightedEnsemble_L2 -0.173585 0.224728 16.698301 6.104501 116.465468 0.004791 0.000365 0.260946 2 True 8
4 LightGBMLarge_BAG_L1 -0.180891 0.191257 19.303928 5.934158 182.045549 19.303928 5.934158 182.045549 1 True 7
5 RandomForestMSE_BAG_L1 -0.218001 0.195891 0.080031 0.118783 2.292897 0.080031 0.118783 2.292897 1 True 5
6 KNeighborsDist_BAG_L1 -0.428627 -0.177638 0.328730 0.138565 0.478930 0.328730 0.138565 0.478930 1 True 2
7 KNeighborsUnif_BAG_L1 -0.452851 -0.163089 0.245754 0.142788 0.494972 0.245754 0.142788 0.494972 1 True 1
model score_val pred_time_val fit_time pred_time_val_marginal fit_time_marginal stack_level can_infer fit_order
0 WeightedEnsemble_L2 0.224728 6.104501 116.465468 0.000365 0.260946 2 True 8
1 LightGBM_BAG_L1 0.223466 5.985353 113.911626 5.985353 113.911626 1 True 4
2 RandomForestMSE_BAG_L1 0.195891 0.118783 2.292897 0.118783 2.292897 1 True 5
3 LightGBMLarge_BAG_L1 0.191257 5.934158 182.045549 5.934158 182.045549 1 True 7
4 LightGBMXT_BAG_L1 0.184157 5.652621 110.592622 5.652621 110.592622 1 True 3
5 ExtraTreesMSE_BAG_L1 0.159837 0.108956 2.480520 0.108956 2.480520 1 True 6
6 KNeighborsUnif_BAG_L1 -0.163089 0.142788 0.494972 0.142788 0.494972 1 True 1
7 KNeighborsDist_BAG_L1 -0.177638 0.138565 0.478930 0.138565 0.478930 1 True 2
on test {'r2': -0.1735846713973992, 'root_mean_squared_error': -0.5873886626250626, 'mean_squared_error':
-0.3450254409804597, 'mean_absolute_error': -0.5025806836078042, 'pearsonr': 0.03216134016875802, '
median_absolute_error': -0.46859056949615474}
-------------------- 2 ------------------------
load y_index:2, y:HDL
0 1.74
1 1.73
2 1.90
3 0.96
4 1.51
...
90 1.63
91 1.82
92 1.88
93 1.90
94 2.08
Name: HDL, Length: 95, dtype: float64
model score_test score_val pred_time_test pred_time_val fit_time pred_time_test_marginal pred_time_val_marginal
fit_time_marginal stack_level can_infer fit_order
0 WeightedEnsemble_L2 0.156421 0.362223 16.139252 6.402312 120.245155 0.004135 0.000346 0.170376 2 True 8
1 ExtraTreesMSE_BAG_L1 0.130413 0.353784 0.254342 0.117741 1.877276 0.254342 0.117741 1.877276 1 True 6
2 RandomForestMSE_BAG_L1 0.118818 0.338788 0.447962 0.112499 2.793362 0.447962 0.112499 2.793362 1 True 5
3 LightGBM_BAG_L1 0.106899 0.324773 15.432814 6.171726 115.404141 15.432814 6.171726 115.404141 1 True 4
4 LightGBMXT_BAG_L1 0.089015 0.278155 14.454313 6.100674 118.794655 14.454313 6.100674 118.794655 1 True 3
5 LightGBMLarge_BAG_L1 0.055995 0.231043 17.744295 5.390294 178.637938 17.744295 5.390294 178.637938 1 True 7
6 KNeighborsUnif_BAG_L1 -0.433167 -0.120092 0.262643 0.161999 0.506232 0.262643 0.161999 0.506232 1 True 1
7 KNeighborsDist_BAG_L1 -0.447588 -0.106983 0.178133 0.127185 0.518703 0.178133 0.127185 0.518703 1 True 2
model score_val pred_time_val fit_time pred_time_val_marginal fit_time_marginal stack_level can_infer fit_order
0 WeightedEnsemble_L2 0.362223 6.402312 120.245155 0.000346 0.170376 2 True 8
1 ExtraTreesMSE_BAG_L1 0.353784 0.117741 1.877276 0.117741 1.877276 1 True 6
2 RandomForestMSE_BAG_L1 0.338788 0.112499 2.793362 0.112499 2.793362 1 True 5
3 LightGBM_BAG_L1 0.324773 6.171726 115.404141 6.171726 115.404141 1 True 4
4 LightGBMXT_BAG_L1 0.278155 6.100674 118.794655 6.100674 118.794655 1 True 3
5 LightGBMLarge_BAG_L1 0.231043 5.390294 178.637938 5.390294 178.637938 1 True 7
6 KNeighborsDist_BAG_L1 -0.106983 0.127185 0.518703 0.127185 0.518703 1 True 2
7 KNeighborsUnif_BAG_L1 -0.120092 0.161999 0.506232 0.161999 0.506232 1 True 1
on test {'r2': 0.15642147141903617, 'root_mean_squared_error': -0.36003092515600815, 'mean_squared_error':
-0.12962226706869115, 'mean_absolute_error': -0.28725089700598466, 'pearsonr': 0.4759554908108065, '
median_absolute_error': -0.24015010833740247}
-------------------- 3 ------------------------
load y_index:3, y:LDL
0 0.46
1 0.46
2 0.48
3 0.22
4 0.46
...
90 0.83
91 0.68
92 0.65
93 0.83
94 0.96
Name: LDL, Length: 95, dtype: float64
model score_test score_val pred_time_test pred_time_val fit_time pred_time_test_marginal pred_time_val_marginal
fit_time_marginal stack_level can_infer fit_order
0 ExtraTreesMSE_BAG_L1 0.583786 0.451255 0.184573 0.116423 3.262111 0.184573 0.116423 3.262111 1 True 6
1 WeightedEnsemble_L2 0.574419 0.451461 0.461115 0.653492 4.007153 0.007372 0.000380 0.168956 2 True 8
2 RandomForestMSE_BAG_L1 0.527375 0.410754 0.346984 0.155024 4.129766 0.346984 0.155024 4.129766 1 True 5
3 LightGBMXT_BAG_L1 0.211455 0.214797 16.309113 9.081986 163.133652 16.309113 9.081986 163.133652 1 True 3
4 LightGBM_BAG_L1 0.183617 0.178562 21.454024 7.527742 159.444175 21.454024 7.527742 159.444175 1 True 4
5 LightGBMLarge_BAG_L1 0.089422 0.141927 18.111882 8.720635 297.492249 18.111882 8.720635 297.492249 1 True 7
6 KNeighborsUnif_BAG_L1 -0.258886 0.121252 0.209997 0.528614 0.725927 0.209997 0.528614 0.725927 1 True 1
7 KNeighborsDist_BAG_L1 -0.286025 0.140641 0.269171 0.536690 0.576085 0.269171 0.536690 0.576085 1 True 2
model score_val pred_time_val fit_time pred_time_val_marginal fit_time_marginal stack_level can_infer fit_order
0 WeightedEnsemble_L2 0.451461 0.653492 4.007153 0.000380 0.168956 2 True 8
1 ExtraTreesMSE_BAG_L1 0.451255 0.116423 3.262111 0.116423 3.262111 1 True 6
2 RandomForestMSE_BAG_L1 0.410754 0.155024 4.129766 0.155024 4.129766 1 True 5
3 LightGBMXT_BAG_L1 0.214797 9.081986 163.133652 9.081986 163.133652 1 True 3
4 LightGBM_BAG_L1 0.178562 7.527742 159.444175 7.527742 159.444175 1 True 4
5 LightGBMLarge_BAG_L1 0.141927 8.720635 297.492249 8.720635 297.492249 1 True 7
6 KNeighborsDist_BAG_L1 0.140641 0.536690 0.576085 0.536690 0.576085 1 True 2
7 KNeighborsUnif_BAG_L1 0.121252 0.528614 0.725927 0.528614 0.725927 1 True 1
on test {'r2': 0.574419016605586, 'root_mean_squared_error': -0.132978135735581, 'mean_squared_error':
-0.01768318458371061, 'mean_absolute_error': -0.1092170268610904, 'pearsonr': 0.7915238869869119, '
median_absolute_error': -0.09634080648422239}
-------------------- 4 ------------------------
load y_index:4, y:APOB
0 0.63
1 0.55
2 0.86
3 0.51
4 0.81
...
90 0.96
91 0.50
92 0.60
93 0.52
94 0.52
Name: APOB, Length: 95, dtype: float64
model score_test score_val pred_time_test pred_time_val fit_time pred_time_test_marginal pred_time_val_marginal
fit_time_marginal stack_level can_infer fit_order
0 RandomForestMSE_BAG_L1 0.008701 0.112709 0.170211 0.123355 2.625814 0.170211 0.123355 2.625814 1 True 5
1 WeightedEnsemble_L2 -0.008198 0.118765 16.298637 6.658705 119.657388 0.003897 0.000428 0.167916 2 True 8
2 ExtraTreesMSE_BAG_L1 -0.038202 0.105381 0.117951 0.192711 2.962204 0.117951 0.192711 2.962204 1 True 6
3 LightGBM_BAG_L1 -0.153034 -0.001656 16.006577 6.342211 113.901454 16.006577 6.342211 113.901454 1 True 4
4 LightGBMLarge_BAG_L1 -0.167203 -0.012781 17.248877 5.675922 175.234315 17.248877 5.675922 175.234315 1 True 7
5 LightGBMXT_BAG_L1 -0.170120 -0.018338 18.537021 5.972352 113.346405 18.537021 5.972352 113.346405 1 True 3
6 KNeighborsDist_BAG_L1 -0.705394 -0.253372 0.134763 0.353030 0.528286 0.134763 0.353030 0.528286 1 True 2
7 KNeighborsUnif_BAG_L1 -0.706369 -0.244869 0.309474 0.219925 1.093336 0.309474 0.219925 1.093336 1 True 1
model score_val pred_time_val fit_time pred_time_val_marginal fit_time_marginal stack_level can_infer fit_order
0 WeightedEnsemble_L2 0.118765 6.658705 119.657388 0.000428 0.167916 2 True 8
1 RandomForestMSE_BAG_L1 0.112709 0.123355 2.625814 0.123355 2.625814 1 True 5
2 ExtraTreesMSE_BAG_L1 0.105381 0.192711 2.962204 0.192711 2.962204 1 True 6
3 LightGBM_BAG_L1 -0.001656 6.342211 113.901454 6.342211 113.901454 1 True 4
4 LightGBMLarge_BAG_L1 -0.012781 5.675922 175.234315 5.675922 175.234315 1 True 7
5 LightGBMXT_BAG_L1 -0.018338 5.972352 113.346405 5.972352 113.346405 1 True 3
6 KNeighborsUnif_BAG_L1 -0.244869 0.219925 1.093336 0.219925 1.093336 1 True 1
7 KNeighborsDist_BAG_L1 -0.253372 0.353030 0.528286 0.353030 0.528286 1 True 2
on test {'r2': -0.008197772208481835, 'root_mean_squared_error': -0.2024248960540698, 'mean_squared_error':
-0.04097583854250096, 'mean_absolute_error': -0.1523386311531067, 'pearsonr': 0.4853830029789988, '
median_absolute_error': -0.10489417076110841}
