// Copyright 2017 Netherlands Institute for Radio Astronomy (ASTRON)
// Copyright 2017 Netherlands eScience Center
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <BeamDriver.hpp>


void generateBeamDriver(bool subbandDedispersion, AstroData::Observation & observation, std::vector< uint8_t > & beamDriver, unsigned int padding) {
  // This driver is just for testing.
  // All subbands/channels will use the same beam, and all beams will be used in a circular way.
  for ( unsigned int beam = 0; beam < observation.getNrSynthesizedBeams(); beam++ ) {
    if ( subbandDedispersion) {
      for ( unsigned int subband = 0; subband < observation.getNrSubbands(); subband++ ) {
        beamDriver[(beam * observation.getNrSubbands(padding / sizeof(uint8_t))) + subband] = beam % observation.getNrBeams();
      }
    } else {
      for ( unsigned int channel = 0; channel < observation.getNrChannels(); channel++ ) {
        beamDriver[(beam * observation.getNrChannels(padding / sizeof(uint8_t))) + channel] = beam % observation.getNrBeams();
      }
    }
  }
}

