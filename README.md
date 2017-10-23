
# AMBER: a Fast Radio Burst real-time pipeline

AMBER is a many-core accelerated and fully auto-tuned pipeline for detecting Fast Radio Bursts and other single pulse transients.

## Publications

* Alessio Sclocco, Joeri van Leeuwen, Henri E. Bal, Rob V. van Nieuwpoort. _A Real-Time Radio Transient Pipeline for ARTS_. **3rd IEEE Global Conference on Signal & Information Processing**, December 14-16, 2015, Orlando (Florida), USA. ([print](http://ieeexplore.ieee.org/xpl/freeabs_all.jsp?arnumber=7418239&abstractAccess=no&userType=inst)) ([preprint](http://alessio.sclocco.eu/pubs/sclocco2015a.pdf)) ([slides](http://alessio.sclocco.eu/pubs/Presentation_GlobalSIP2015.pdf))

## Dependencies

* [utils](https://github.com/isazi/utils) - master branch
* [OpenCL](https://github.com/isazi/OpenCL) - master branch
* [AstroData](https://github.com/AA-ALERT/AstroData) - master branch
* [Dedispersion](https://github.com/isazi/Dedispersion) - master branch
* [Integration](https://github.com/isazi/Integration) - master branch
* [SNR](https://github.com/isazi/SNR) - master branch



./bin/TransientSearch
-opencl_platform ...
-opencl_device ...

-device_name allows you to use predefined configs from the input files

Files made by tuning the separate parts (ie. output of tuning / analyse scripts and some editing)
-padding_file ...
-dedispersion_file ...
-integration_file ...
-snr_file ...

-zapped_channels ...     see Dedispersion
-integration_steps ...   see Integration

[-print]                 Prints extra info
[-compact_results]       Merge consecutive candidates into a single one
-input_bits ...          see Dedispersion
-output ...              Output file name
-dm_node ...             Number of DMs per node (split via MPI)
-dm_first ...            see Desispersion
-dm_step ...             see Desispersion
-threshold ...           signal to noise threshold

Data sources:
Lofar:
-lofar -header ...  -data ...  [-limit] -limit -seconds ...

Sigproc:
-sigproc -header ...  -data ...  -seconds ...  -channels ...  -min_freq ...  -channel_bandwidth ...  -samples ...

PSR DADA ring buffer:
-dada -dada_key ... -beams ... -seconds ...

default, Generated data:
[-random] -width ... -dm ... -beams ... -seconds ... -channels ... -min_freq ... -channel_bandwidth ... -samples ...


## License

Licensed under the Apache License, Version 2.0.

