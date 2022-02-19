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

#define BMO_ENABLE_STATS_FEATURES

#include "bmo_tests.hpp"

int main()
{
    const int n_vals = 4;

    bmo_stats::rand_engine_t engine(1);

    //

    ColVec_t runif_test_vec = bmo_stats::runif_vec<double>(n_vals, engine);
    
    BMO_MATOPS_COUT << runif_test_vec << BMO_MATOPS_ENDL;

    ColVec_t rnorm_test_vec = bmo_stats::rnorm_vec<double>(n_vals, engine);
    
    BMO_MATOPS_COUT << rnorm_test_vec << BMO_MATOPS_ENDL;

    ColVec_t rind_test_vec = bmo_stats::rind_vec(n_vals, 0, 3, engine);
    
    BMO_MATOPS_COUT << rind_test_vec << BMO_MATOPS_ENDL;

    return 0;
}
