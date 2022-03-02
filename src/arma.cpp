#include <iostream>

#include <armadillo>


int main() {

    arma::mat m(100, 3);
    m.transform([](double){return arma::randn();});

    arma::mat U,V;
    arma::vec S;
    arma::svd(U,S,V,m);

    std::cout << S.tail(1)[0] * S.tail(1)[0] << std::endl;
    std::cout << arma::accu((m * V.tail_cols(1)) % (m * V.tail_cols(1))) << std::endl;
}