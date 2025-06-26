for date in 20250625

do 
    base_url="https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr"
    year_month=${date:0:6}

    # SST is only tracked once per day, so no time variable
    std_file="oisst-avhrr-v02r01.${date}.nc"
    prelim_file="oisst-avhrr-v02r01.${date}_preliminary.nc"
    out_file="sst_data/sst_$date"

    wget --no-check-certificate -O "$out_file" "$base_url/$year_month/$std_file" || \
    wget --no-check-certificate -O "$out_file" "$base_url/$year_month/$prelim_file" || {
        echo "Error: Could not download SST file for $date (standard or preliminary)."
        exit 1
    }

    echo "SST collected for $date"
done
