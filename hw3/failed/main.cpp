#include "matwrapper.h"

int main(int argc, char** argv)
{
    init(&argc, &argv);
    WrapperMatNative<NRA, NCA, NCB> a;
    std::cout <<NRA<<'\n';

    a.init();
    /* a.simpleMult(); */

    a.init();
    a.openmpMult(1);

    cleanup();
    return 0;
    a.init();
    a.mpiMult();
    /* a.printC(); */

    cleanup();
    return 0;
}
