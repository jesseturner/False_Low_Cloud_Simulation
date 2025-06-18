for date in 20250612

do 
	#--- SST is only tracked once per day, so no time variable
    sst_path=https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/${date:0:6}/oisst-avhrr-v02r01."$date".nc

    wget --no-check-certificate -O sst_data/sst_$date $sst_path
    echo "SST collected for "$date

done