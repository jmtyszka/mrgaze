#include "mrgaze.h"
#include "ui_mrgaze.h"

MrGaze::MrGaze(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MrGaze)
{
    ui->setupUi(this);
}

MrGaze::~MrGaze()
{
    delete ui;
}

void MrGaze::on_Glint_Thresh_Dial_valueChanged(int value)
{

}
