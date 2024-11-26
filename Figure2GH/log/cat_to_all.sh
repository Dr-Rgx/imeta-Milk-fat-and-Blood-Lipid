interpretable_or_best_quality=best_quality
for i in {0..20}
  do

  prefix=autogluon_res/stage$i/$interpretable_or_best_quality/


cat ${prefix}r2_on_all_y_i_0__y_name_CHO.txt    enter   \
    ${prefix}r2_on_all_y_i_1__y_name_TG.txt     enter   \
    ${prefix}r2_on_all_y_i_2__y_name_HDL.txt    enter   \
    ${prefix}r2_on_all_y_i_3__y_name_LDL.txt    enter   \
    ${prefix}r2_on_all_y_i_4__y_name_APOB.txt   enter   \
    > ${prefix}all.txt

    echo catfineshed $prefix

  done



