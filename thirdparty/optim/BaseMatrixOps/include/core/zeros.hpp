/*################################################################################
  ##
  ##   Copyright (C) 2016-2022 Keith O'Hara
  ##
  ##   This file is part of the BaseMatrixOps C++ library.
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

/*
 * zeros
 */

#ifndef BMO_MATOPS_ZEROS_VEC

//

#ifdef BMO_ENABLE_ARMA_WRAPPERS
    #define BMO_MATOPS_ZERO_COLVEC(n) ColVec_t(n,arma::fill::zeros)
    #define BMO_MATOPS_ZERO_ROWVEC(n) RowVec_t(n,farma::ill::zeros)
    #define BMO_MATOPS_ZERO_MAT(n,k) Mat_t(n,k,arma::fill::zeros)
#endif

#ifdef BMO_ENABLE_EIGEN_WRAPPERS
    // #define BMO_MATOPS_ZERO_COLVEC(n) Eigen::VectorXd::Zero(n)
    #define BMO_MATOPS_ZERO_COLVEC(n) ColVec_t::Zero(n)
    #define BMO_MATOPS_ZERO_ROWVEC(n) ColVec_t::Zero(n).transpose()
    // #define BMO_MATOPS_ZERO_MAT(n,k) Eigen::MatrixXd::Zero(n,k)
    #define BMO_MATOPS_ZERO_MAT(n,k) Mat_t::Zero(n,k)
#endif

//

#endif
