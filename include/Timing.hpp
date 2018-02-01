// Copyright 2018 Netherlands Institute for Radio Astronomy (ASTRON)
// Copyright 2018 Netherlands eScience Center
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

#pragma once

#include "configuration.hpp"

struct Timers {
  isa::utils::Timer inputLoad;
  isa::utils::Timer search;
  isa::utils::Timer inputHandling;
  isa::utils::Timer inputCopy;
  isa::utils::Timer dedispersionSingleStep;
  isa::utils::Timer dedispersionStepOne;
  isa::utils::Timer dedispersionStepTwo;
  isa::utils::Timer integration;
  isa::utils::Timer snr;
  isa::utils::Timer outputCopy;
  isa::utils::Timer trigger;
};
