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

#include <vector>
#include <map>

#include <Observation.hpp>

#pragma once

struct triggeredEvent {
  unsigned int beam = 0;
  unsigned int sample = 0;
  unsigned int integration = 0;
  float DM = 0.0f;
  float SNR = 0.0f;
};

struct compactedEvent : triggeredEvent {
  unsigned int compactedDMs = 1;
};

using triggeredEvents_t = std::vector<std::map<unsigned int, std::vector<triggeredEvent>>>;
using compactedEvents_t = std::vector<std::vector<compactedEvent>>;

void trigger(const bool subbandDedispersion, const unsigned int padding, const unsigned int integration, const float threshold, const AstroData::Observation & observation, const std::vector<float> & snrData, const std::vector<unsigned int> & samplesData, triggeredEvents_t & triggeredEvents);
void compact(const AstroData::Observation & observation, const triggeredEvents_t & triggeredEvents, compactedEvents_t & compactedEvents);
