b="2014-11-29 11:00"
e="2014-11-29 23:59"

echo Plotting from $b till $e 
echo If you want different dates, please edit this script
echo

syncdaq_get -b "$b" -e "$e" xfel_bl_3_st_4_motor_29/position | sed 's:pulse::g' > delay.txt
syncdaq_get -b "$b" -e "$e" xfel_bl_3_tc_mono_1_theta/position | sed 's:pulse::g' > mono.txt
syncdaq_get -b "$b" -e "$e" xfel_bl_3_st_4_motor_27/position | sed 's:pulse::g' > mirr_rot.txt
syncdaq_get -b "$b" -e "$e" xfel_bl_3_st_4_motor_28/position | sed 's:pulse::g' > mirr_tilt.txt
syncdaq_get -b "$b" -e "$e" xfel_bl_3_st_4_motor_26/position | sed 's:pulse::g' > ND.txt
python check_scans.py -p delay.txt mono.txt mirr_rot.txt mirr_tilt.txt ND.txt

