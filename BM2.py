import numpy as np
from scipy.linalg import lu, solve

#Для воспроизводительности результатов устанавливаем seed
np.random.seed(42)

#Задание 1
#Создать квадратную матрицу из случайных вещественных чисел из
#(2,4) размера 8 . Найти скалярное произведение 3 строки на 8 столбец.
#Использовать срезы матриц.

matrix_a1 = np.random.uniform(2, 4, size = (8,8))
print(f"Матрица А1 (8x8):\n{matrix_a1}\n")

row_3 = matrix_a1[2,:]
col_8 = matrix_a1[:,7]

scalar_res = np.sum(row_3 * col_8)
print(f"3-я строка матрицы: {row_3}")
print(f"8-ой столбец матрицы: {col_8}")
print(f"Скалярное произведение 3-й строки на 8-й столбец: {scalar_res:.4f}")

#Задание 2
#Создать две матрицы из случайных целых чисел из отрезка [-2,6]
#подходящего размера. Найти их произведение тремя способами: 1)
#записав скалярный алгоритм умножения матриц 2) записав векторный
#алгоритм, записав матрицу С 3) проверив с помощью функции np.dot.

m,k,n = 4,5,3

matrix_a2 = np.random.randint(-2, 7, size = (m,k))
matrix_b2 = np.random.randint(-2, 7, size = (k,n))

print(f"Матрица А2 ({m}x{k}):\n{matrix_a2}\n")
print(f"Матрица B2 ({k}x{n}):\n{matrix_b2}\n")

#Скалярный алгоритм
def matrix_multiply_scalar(A,B):
    m,k = A.shape
    k2,n =  B.shape
    C = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            for t in range(k):
                C[i, j] += A[i,t] * B[t,j]
    return C

def matrix_multiply_vector(A,B):
    m, k = A.shape
    k2, n = B.shape
    C = np.zeros((m,n))

    for i in range(m):
        for j in range(n):
            for t in range(k):
                C[i, j] = np.dot(A[i,:], B[:,j])
    return C

C_scalar = matrix_multiply_scalar(matrix_a2, matrix_b2)
print("Результат скалярного алгоритма:\n", C_scalar)

C_vector = matrix_multiply_vector(matrix_a2, matrix_b2)
print("\nРезультат векторного алгоритма:\n", C_vector)

C_dot = np.dot(matrix_a2, matrix_b2)
print("\nРезультат np.dot:\n", C_dot)

#Задание 3
#Создать произвольную нижнетреугольную матрицу А 5 порядка (не
#унитреугольную), вектор B произвольный. Решить систему AX = B.

matrix_a3 = np.tril(np.random.uniform(1, 5, size = (5,5)))
np.fill_diagonal(matrix_a3, np.random.uniform(2, 5, size = 5))

print(f"Нижнетреугольная матрица A (5x5):\n{matrix_a3}\n")

matrix_b3 = np.random.uniform(1, 10, size = 5)
print(f"Вектор B:\n{matrix_b3}\n")

def solve_lower_triangular(L,b):
    n = len(b)
    x = np.zeros(n)
    for i in range(n):
        sum_val = np.dot(L[i,:i], x[:i])
        x[i] = (b[i] - sum_val) / L[i,i]
    return x

X3 = solve_lower_triangular(matrix_a3, matrix_b3)
print(f"Решение системы X: {X3}\n")

X3_check = np.linalg.solve(matrix_a3, matrix_b3)
print(f"Решение системы X: {X3_check}\n")
print(f"Невязка:{np.linalg.norm(matrix_a3 @ X3_check - matrix_b3)}")
print(f"Совпадение решений: {np.allclose(X3, X3_check)}")

#Задание 4
#Решить систему, используя LU разложение

matrix_a4 = np.random.uniform(1, 5, size = (5,5))
matrix_b4 = np.random.uniform(1, 10, size = 5)
print(f"Матрица А4 (5х5):\n{matrix_a4}\n")
print(f"Матрица B4 (5x5):\n{matrix_b4}\n)")

#lu возвращает P, L, U такие, что A = P^T @ L @ U
P, L, U = lu(matrix_a4)
print("Матрица L:\n", L)
print("\nМатрица U:\n", U)
print("Матрица перестановок P:\n", P)

B_permuted = P.T @ matrix_b4

def solve_lower_unit(L, b):
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i,:i], y[:i])
    return y

def solve_upper(U, y):
    n = len(y)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i,i+1:], x[i+1:])) / U[i,i]
    return x

#Ly = PB
Y = solve_lower_unit(L, B_permuted)
print(f"\nПромежуточный вектор Y (решение Ly = PB):\n{Y}\n")

#Ux = Y
X4 = solve_upper(U, Y)
print(f"Итоговое решение X (LU-разложение):\n{X4}\n")

X4_check = solve(matrix_a4, matrix_b4)
print(f"Решение системы X (scipy.solve):\n{X4_check}\n")

#Проверка невязки
residual = np.linalg.norm(matrix_a4 @ X4 - matrix_b4)
print(f"\nНевязка ||AX - B||: {residual:.2e}")
print(f"Совпадение с эталонным решением: {np.allclose(X4, X4_check)}")
