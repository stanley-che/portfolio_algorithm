#include <iostream>
#include "prediction.h"

int main() {
    TrainConfig cfg;
    std::cout << cfg.window << " " << cfg.half_life << "\n";
    cfg.model = ModelType::Ridge;
    return 0;
}

