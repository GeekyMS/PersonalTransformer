#include "tape.h"

std::vector<Node> tape;

void tape_push(std::function<void()> backward) {
    tape.push_back(Node{backward});
}

void tape_backward() {
    for(int i = (int)tape.size() - 1; i >= 0; i--){
        tape[i].backward();
    }
}

void tape_reset() {
    tape.clear();
}
