#! /usr/bin/bash

for date in 20250612

do 
	#--- must be 00z, 06z, 12z, 18z
	dtime=06z

	#--- data can be browsed at: https://noaa-gfs-bdp-pds.s3.amazonaws.com/index.html
	wget -O model_data/gfs_$date https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs."$date"/${dtime:0:2}/atmos/gfs.t"$dtime".pgrb2.0p25.f000
    echo "GFS collected for "$date

done