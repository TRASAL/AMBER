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

#include <Trigger.hpp>


void trigger(const bool subbandDedispersion, const unsigned int padding, const unsigned int integration, const float threshold, const AstroData::Observation & observation, const std::vector<float> & snrData, const std::vector<unsigned int> & samplesData, TriggeredEvents & triggeredEvents) {
  unsigned int nrDMs = 0;

  if ( subbandDedispersion ) {
    nrDMs = observation.getNrDMs(true) * observation.getNrDMs();
  } else {
    nrDMs = observation.getNrDMs();
  }
#pragma omp parallel for
  for ( unsigned int beam = 0; beam < observation.getNrSynthesizedBeams(); beam++ ) {
    for ( unsigned int dm = 0; dm < nrDMs; dm++ ) {
      if ( snrData[(beam * isa::utils::pad(nrDMs, padding / sizeof(float))) + dm] >= threshold ) {
        TriggeredEvent event;
        event.beam = beam;
        event.sample = samplesData[(beam * isa::utils::pad(nrDMs, padding / sizeof(unsigned int))) + dm];
        event.integration = integration;
        event.DM = dm;
        event.SNR = snrData[(beam * isa::utils::pad(nrDMs, padding / sizeof(float))) + dm];
        try {
          // Add event to its existing list
          triggeredEvents.at(beam).at(dm).push_back(event);
        } catch ( std::out_of_range & err ) {
          // Add event to new list
          std::vector<TriggeredEvent> events;

          events.push_back(event);
          triggeredEvents.at(beam).insert(std::pair<unsigned int, std::vector<TriggeredEvent>>(dm, events));
        }
      }
    }
  }
}

void compact(const AstroData::Observation & observation, const TriggeredEvents & triggeredEvents, CompactedEvents & compactedEvents) {
  CompactedEvents temporaryEvents(observation.getNrSynthesizedBeams());

  // Compact integration
  for ( auto beamEvents = triggeredEvents.begin(); beamEvents != triggeredEvents.end(); ++beamEvents ) {
    for ( auto dmEvents = beamEvents->begin(); dmEvents != beamEvents->end(); ++dmEvents ) {
      CompactedEvent event;

      for ( auto dmEvent = dmEvents->second.begin(); dmEvent != dmEvents->second.end(); ++dmEvent ) {
        if ( dmEvent->SNR > event.SNR ) {
          event.beam = dmEvent->beam;
          event.sample = dmEvent->sample;
          event.integration = dmEvent->integration;
          event.DM = dmEvent->DM;
          event.SNR = dmEvent->SNR;
        }
      }
      event.compactedIntegration = dmEvents->second.size();
      temporaryEvents.at(event.beam).push_back(event);
    }
  }
  // Compact DM
  for ( auto beamEvents = temporaryEvents.begin(); beamEvents != temporaryEvents.end(); ++beamEvents ) {
    for ( auto event = beamEvents->begin(); event != beamEvents->end(); ++event ) {
      CompactedEvent finalEvent;
      unsigned int window = 0;

      while ( (event->DM + window) == (event + window)->DM ) {
        if ( (event + window)->SNR > finalEvent.SNR ) {
          finalEvent.beam = (event + window)->beam;
          finalEvent.sample = (event + window)->sample;
          finalEvent.integration = (event + window)->integration;
          finalEvent.compactedIntegration = (event + window)->compactedIntegration;
          finalEvent.DM = (event + window)->DM;
          finalEvent.SNR = (event + window)->SNR;
        }
        window++;
      }
      finalEvent.compactedDMs = window;
      compactedEvents.at(finalEvent.beam).push_back(finalEvent);
      // Move the iterator forward
      event += (window - 1);
    }
  }
}

