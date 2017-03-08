
export SOURCE_ROOT=$HOME/pipeline
mkdir -p "${SOURCE_ROOT}/src"

# Machine specific settings
# DAS5
module load hdf5
export HDF5=/cm/shared/apps/hdf5/current/
export HDF5_INCLUDE="-I${HDF5}/include"
export HDF5_LIBS="-L${HDF5}/lib"
export HDF5_LDFLAGS="-lhdf5 -lhdf5_cpp -lz"

# manually copy psrdada
echo "PSRDada"
cd "${SOURCE_ROOT}/src"
if [ -f ~/psrdada.tar.gz ]; then
  tar -xvf ~/psrdada.tar.gz
  cd psrdada
  make
else
  echo "Please install psrdada in ${SOURCE_ROOT}/src"
fi


module load cuda80/toolkit


# fillringbuffer code
echo "Ringbuffer"
cd "${SOURCE_ROOT}/src"
git clone -b sc4 https://github.com/AA-ALERT/ringbuffer.git

echo "Utils"
cd "${SOURCE_ROOT}/src"
git clone            https://github.com/isazi/utils.git
cd utils && make all

echo "OpenCL"
cd "${SOURCE_ROOT}/src"
git clone            https://github.com/isazi/OpenCL.git
cd OpenCL && make all

echo "AstroData"
cd "${SOURCE_ROOT}/src"
git clone            https://github.com/isazi/AstroData.git
cd AstroData && make all

echo "Dedispersion"
cd "${SOURCE_ROOT}/src"
git clone            https://github.com/isazi/Dedispersion.git
cd Dedispersion && make all

echo "Integration"
cd "${SOURCE_ROOT}/src"
git clone            https://github.com/isazi/Integration.git
cd Integration && make all

echo "SNR"
cd "${SOURCE_ROOT}/src"
git clone            https://github.com/isazi/SNR.git
cd SNR && make all

echo "TransientSearch"
cd "${SOURCE_ROOT}/src"
git clone            https://github.com/isazi/TransientSearch.git
cd TransientSearch && make all

