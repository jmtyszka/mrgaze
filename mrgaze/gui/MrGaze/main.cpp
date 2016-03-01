#include "mrgaze.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MrGaze w;
    w.show();

    return a.exec();
}
