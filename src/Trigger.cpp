// Copyright 2017 ASTRON - Netherlands Institute for Radio Astronomy
// Copyright 2017 NLeSC - Netherlands eScience Center
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

// Contributor: Alessio Sclocco <a.sclocco@esciencecenter.nl>

#include <Trigger.hpp>


void trigger(const bool subbandDedispersion, const unsigned int padding, const unsigned int integration, const float threshold, const AstroData::Observation & observation, const std::vector<float> & snrData, const std::vector<unsigned int> & samplesData, triggeredEvents_t & triggeredEvents) {
  unsigned int nrDMs = 0;

  if ( subbandDedispersion ) {
    nrDMs = observation.getNrDMsSubbanding() * observation.getNrDMs();
  } else {
    nrDMs = observation.getNrDMs();
  }
#pragma omp parallel for
  for ( unsigned int beam = 0; beam < observation.getNrSynthesizedBeams(); beam++ ) {
    for ( unsigned int dm = 0; dm < nrDMs; dm++ ) {
      if ( snrData[(beam * isa::utils::pad(nrDMs, padding / sizeof(float))) + dm] >= threshold ) {
        triggeredEvent event;
        event.beam = beam;
        event.sample = samplesData[(beam * isa::utils::pad(nrDMs, padding / sizeof(unsigned int))) + dm];
        event.integration = integration;
        event.DM = dm;
        event.SNR = snrData[(beam * isa::utils::pad(nrDMs, padding / sizeof(float))) + dm];
        try {
          // Add event to its existing list
          triggeredEvents.at(beam).at(dm).push_back(event);
        } catch ( std::out_of_range err ) {
          // Add event to new list
          std::vector<triggeredEvent> events;

          events.push_back(event);
          triggeredEvents.at(beam).insert(std::pair<unsigned int, std::vector<triggeredEvent>>(dm, events));
        }
      }
    }
  }
}

void compact(const AstroData::Observation & observation, const triggeredEvents_t & triggeredEvents, compactedEvents_t & compactedEvents) {
  compactedEvents_t temporaryEvents(observation.getNrSynthesizedBeams());

  // Compact integration
#pragma omp parallel for
  for ( auto beamEvents = triggeredEvents.begin(); beamEvents != triggeredEvents.end(); ++beamEvents ) {
    auto dmEvents = beamEvents->begin()->second;
    compactedEvent event;

    for ( auto dmEvent = dmEvents.begin(); dmEvent != dmEvents.end(); ++dmEvent ) {
      if ( dmEvent->SNR > event.SNR ) {
        event.beam = dmEvent->beam;
        event.sample = dmEvent->sample;
        event.integration = dmEvent->integration;
        event.DM = dmEvent->DM;
        event.SNR = dmEvent->SNR;
      }
    }
    temporaryEvents.at(event.beam).push_back(event);
  }
  // Compact DM
#pragma omp parallel for
  for ( auto beamEvents = temporaryEvents.begin(); beamEvents != temporaryEvents.end(); ++beamEvents ) {
    for ( auto event = beamEvents->begin(); event != beamEvents->end(); ++event ) {
      compactedEvent finalEvent;
      unsigned int window = 0;

      while ( (event->DM + window) == (event + window)->DM ) {
        finalEvent.beam = event->beam;
        finalEvent.sample = event->sample;
        finalEvent.integration = event->integration;
        finalEvent.DM = event->DM;
        finalEvent.SNR = event->SNR;
        finalEvent.compactedDMs += window;
        window++;
      }
      event += (window - 1);
      compactedEvents.at(finalEvent.beam).push_back(finalEvent);
    }
  }
}

