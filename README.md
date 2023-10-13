# eps-complexity

Algorithm for calculating eps-complexity described by [Dr Piryatinska](https://faculty.sfsu.edu/~alpiryat/) et. al. [here](https://pubmed.ncbi.nlm.nih.gov/29054253/).

We are trying to reproduce results of Holder Exponent estimation by Dubnov Yu.A. et. al. in [this article](https://www.mathnet.ru/php/archive.phtml?wshow=paper&jrnid=at&paperid=16164&option_lang=rus).

First goal - to find estimation of Holder exponent for $\sum \sin(w_i x)$ on some segment $x \in [0, T]$, where $w_i$ are primes. According to theory of eps-complexity (article by [
Boris Darkhovsky](https://www.researchgate.net/publication/359770918_Ocenka_pokazatela_Geldera_na_osnove_koncepcii_epsilon-sloznosti_nepreryvnyh_funkcijEstimate_of_the_Holder_Exponent_Based_on_the_epsilon-Complexity_of_Continuous_Functions)) and computational experiment in article by Dubnov Y.A. et.al. we should get the estimation:  $$\lim_{n\to\infty}\frac{\hat{A_n}}{\log n} = -p = -1.$$


Authors: Dr Alexandra Piryatinska, Daniil Matveev.
