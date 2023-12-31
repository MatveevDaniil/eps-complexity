# Структура логов

1. Логи состоят из набора json файлов, каждый json-файл соотетствует логам просчёта отношения $\frac{-A}{\log(n)}$ для конкретного $n$. 
2. Название каждого json-файла имеет формат - "h=<шаг сетки>.json"
3. Каждый json-файл имеет структуру словаря, в котором __ключи__ - название данных/переменных, __значения__ - значение данных/переменных. Ниже описание полей:
    * h: шаг сетки
    * n: размер сетки ($1/h \pm 1$)
    * t: значение узлов сетки
    * x: значение функции $x(t)=\sum_k \sin{(w_k t)}$, где $w_k$ - простые числа на [241, 293] в узлах сетки
    * (1/drop_ratio, error): error посчитана с использованием сплайнов 1-4 степени и кусочных констант.
    * (log(drop_ratio), log_error): $A, B$ коэффициенты OLS в $\log \epsilon \approx A + B \log(drop\_ratio)$
    * A: обозначается $\hat{A}$ в статьях
    * B: обозначается $\hat{B}$ в статьях
    * -A / log(n): обозначается $p$ в статьях, должен сходится к 1
4. Это первый вариант логов. Скорее всего для сравнения результатов работы наших алгоритмов, придётся добавить логи промежуточных вычеслений. 