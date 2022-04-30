#pragma once
#include <armadillo>

arma::vec4 quat_from_aa(arma::vec3 aa);
arma::vec3 quat_to_aa(arma::vec4 q);
arma::vec4 quat_prod(arma::vec4 p, arma::vec4 q);
arma::vec4 quat_conj(arma::vec4 q);
arma::vec3 quat_rotate_point(arma::vec4 q, arma::vec3 p);
arma::vec4 quat_slerp(arma::vec4 p, arma::vec4 q, double t);
arma::vec4 quat_squad(arma::vec4 p0, arma::vec4 p1, arma::vec4 p2, arma::vec4 p3, double t);
arma::vec4 quat_quad(arma::vec4 p0, arma::vec4 p1, arma::vec4 p2, arma::vec4 p3, double t);