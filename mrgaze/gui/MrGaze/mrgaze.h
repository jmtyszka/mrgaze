#ifndef MRGAZE_H
#define MRGAZE_H

#include <QMainWindow>

namespace Ui {
class MrGaze;
}

class MrGaze : public QMainWindow
{
    Q_OBJECT

public:
    explicit MrGaze(QWidget *parent = 0);
    ~MrGaze();

private:
    Ui::MrGaze *ui;
};

#endif // MRGAZE_H
