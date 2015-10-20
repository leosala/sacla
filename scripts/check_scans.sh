#!/bin/bash

b=''
e=''

HELP="""\nThis script runs syncdaq_get, produces text files, and run check_scans.py to produce nice plots of DAQ quantities.\n\n

USAGE: $0 --begin 'begin_of_time_range' --end 'end_of_time_range'
\n\n
EXAMPLES:\n
      $0 --begin '2014-11-29 11:00:00' --end '2014-11-29 23:59:00'\n
      $0 -b '2014-11-29 11:00:00' -e '2014-11-29 23:59:00'\n
"""

while [[ $# > 1 ]]; do
    key="$1"
    
    case $key in
	-b|--begin)
	    b="$2"
	    shift # past argument
	    ;;
	-e|--end)
	    e="$2"
	    shift # past argument
	    ;;
	-h|--help)
	    echo $HELP
	    exit
	    ;;
	*)
            # unknown option
	    ;;
    esac
    shift # past argument or value
done

if [ "$b" == '' ]; then
    echo -e $HELP
    exit
fi

echo Plotting from $b till $e 
echo

syncdaq_get -b "$b" -e "$e" xfel_bl_3_st_4_motor_29/position | sed 's:pulse::g' > delay.txt
syncdaq_get -b "$b" -e "$e" xfel_bl_3_tc_mono_1_theta/position | sed 's:pulse::g' > mono.txt
syncdaq_get -b "$b" -e "$e" xfel_bl_3_st_4_motor_27/position | sed 's:pulse::g' > mirr_rot.txt
syncdaq_get -b "$b" -e "$e" xfel_bl_3_st_4_motor_28/position | sed 's:pulse::g' > mirr_tilt.txt
syncdaq_get -b "$b" -e "$e" xfel_bl_3_st_4_motor_26/position | sed 's:pulse::g' > ND.txt
python check_scans.py -b "$b" -e "$e" -p delay.txt mono.txt mirr_rot.txt mirr_tilt.txt ND.txt

