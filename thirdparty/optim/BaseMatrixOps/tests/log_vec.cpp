/*################################################################################
  ##
  ##   Copyright (C) 2016-2022 Keith O'Hara
  ##
  ##   This file is part of the GCE-Math C++ library.
  ##
  ##   Licensed under the Apache License, Version 2.0 (the "License");
  ##   you may not use this file except in compliance with the License.
  ##   You may obtain a copy of the License at
  ##
  ##       http://www.apache.org/licenses/LICENSE-2.0
  ##
  ##   Unless required by applicable law or agreed to in writing, software
  ##   distributed under the License is distributed on an "AS IS" BASIS,
  ##   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ##   See the License for the specific language governing permissions and
  ##   limitations under the License.
  ##
  ################################################################################*/

#include "bmo_tests.hpp"

int main()
{
    const int n_vals = 4;

    ColVec_t test_vec = BMO_MATOPS_ONE_COLVEC(n_vals);
    
    BMO_MATOPS_COUT << test_vec << BMO_MATOPS_ENDL;

    test_vec = BMO_MATOPS_LOG(test_vec);

    BMO_MATOPS_COUT << test_vec << BMO_MATOPS_ENDL;

    test_vec(n_vals - 1) = -1;

    BMO_MATOPS_COUT << test_vec << BMO_MATOPS_ENDL;

    return 0;
}
