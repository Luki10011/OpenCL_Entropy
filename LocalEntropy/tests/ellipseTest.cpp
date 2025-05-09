#include <iostream>
#include <vector>

/**
* Generate structure element - m x n ellipse as 1D vector
*/
std::vector<u_char> generateEllipse(int m, int n) {
    std::vector<u_char> mask(m * n, 0);

    float a = (n-1) / 2.0f;              
    float b = (m-1) / 2.0f;           
    float x0 = a;
    float y0 = b;

    for (int y = 0; y < m; ++y) {
        for (int x = 0; x < n; ++x) {
            float dx = (x - x0) / a;
            float dy = (y - y0) / b;
            if (dx * dx + dy * dy <= 1.0f)
                mask[y * n + x] = 1;
        }
    }

    return mask;
}
// Funkcja do wypisania maski w postaci 2D
void printMask(const std::vector<u_char>& mask, int m, int n) {
    for (int y = 0; y < m; ++y) {
        for (int x = 0; x < n; ++x) {
            std::cout << (int)mask[y * n + x] << " ";  // Wypisujemy wartość 0 lub 1
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Funkcja testująca
void testEllipses() {
    // Przykładowe rozmiary masek do testowania
    std::vector<std::pair<int, int>> testCases = {
        {4, 6},
        {5, 10},
        {6, 8},
        {7, 7},
        {10, 15}
    };

    for (auto& testCase : testCases) {
        int m = testCase.first;
        int n = testCase.second;

        std::cout << "Testing ellipse for mask size " << m << "x" << n << std::endl;

        // Generowanie maski elipsy
        std::vector<u_char> mask = generateEllipse(m, n);

        // Wypisanie maski
        printMask(mask, m, n);
    }
}

int main() {
    // Uruchamiamy testy
    testEllipses();
    return 0;
}