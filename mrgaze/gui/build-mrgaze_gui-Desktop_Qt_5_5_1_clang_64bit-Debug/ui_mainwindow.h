/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.5.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QDial>
#include <QtWidgets/QFrame>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actionOpen;
    QAction *actionQuit;
    QAction *actionAbout_MrGaze;
    QWidget *centralWidget;
    QFrame *frame;
    QFrame *frame_2;
    QTabWidget *tabWidget;
    QWidget *AdjustTab;
    QDial *dial;
    QDial *dial_2;
    QLineEdit *lineEdit;
    QLineEdit *lineEdit_2;
    QWidget *Calibration_Tab;
    QWidget *Tracking_Tab;
    QMenuBar *menuBar;
    QMenu *menuGaze_Estimation;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(1080, 711);
        actionOpen = new QAction(MainWindow);
        actionOpen->setObjectName(QStringLiteral("actionOpen"));
        actionQuit = new QAction(MainWindow);
        actionQuit->setObjectName(QStringLiteral("actionQuit"));
        actionAbout_MrGaze = new QAction(MainWindow);
        actionAbout_MrGaze->setObjectName(QStringLiteral("actionAbout_MrGaze"));
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        frame = new QFrame(centralWidget);
        frame->setObjectName(QStringLiteral("frame"));
        frame->setGeometry(QRect(20, 20, 512, 384));
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        frame_2 = new QFrame(centralWidget);
        frame_2->setObjectName(QStringLiteral("frame_2"));
        frame_2->setGeometry(QRect(550, 20, 512, 384));
        frame_2->setFrameShape(QFrame::StyledPanel);
        frame_2->setFrameShadow(QFrame::Raised);
        tabWidget = new QTabWidget(centralWidget);
        tabWidget->setObjectName(QStringLiteral("tabWidget"));
        tabWidget->setGeometry(QRect(20, 420, 1041, 221));
        AdjustTab = new QWidget();
        AdjustTab->setObjectName(QStringLiteral("AdjustTab"));
        dial = new QDial(AdjustTab);
        dial->setObjectName(QStringLiteral("dial"));
        dial->setGeometry(QRect(10, 0, 101, 101));
        dial->setMaximum(100);
        dial_2 = new QDial(AdjustTab);
        dial_2->setObjectName(QStringLiteral("dial_2"));
        dial_2->setGeometry(QRect(110, 0, 101, 101));
        dial_2->setMaximum(100);
        dial_2->setValue(100);
        lineEdit = new QLineEdit(AdjustTab);
        lineEdit->setObjectName(QStringLiteral("lineEdit"));
        lineEdit->setGeometry(QRect(20, 100, 81, 21));
        lineEdit->setAlignment(Qt::AlignCenter);
        lineEdit_2 = new QLineEdit(AdjustTab);
        lineEdit_2->setObjectName(QStringLiteral("lineEdit_2"));
        lineEdit_2->setGeometry(QRect(120, 100, 81, 21));
        lineEdit_2->setAlignment(Qt::AlignCenter);
        tabWidget->addTab(AdjustTab, QString());
        Calibration_Tab = new QWidget();
        Calibration_Tab->setObjectName(QStringLiteral("Calibration_Tab"));
        tabWidget->addTab(Calibration_Tab, QString());
        Tracking_Tab = new QWidget();
        Tracking_Tab->setObjectName(QStringLiteral("Tracking_Tab"));
        tabWidget->addTab(Tracking_Tab, QString());
        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1080, 22));
        menuGaze_Estimation = new QMenu(menuBar);
        menuGaze_Estimation->setObjectName(QStringLiteral("menuGaze_Estimation"));
        MainWindow->setMenuBar(menuBar);
        mainToolBar = new QToolBar(MainWindow);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        MainWindow->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainWindow->setStatusBar(statusBar);

        menuBar->addAction(menuGaze_Estimation->menuAction());

        retranslateUi(MainWindow);

        tabWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", 0));
        actionOpen->setText(QApplication::translate("MainWindow", "Open", 0));
        actionQuit->setText(QApplication::translate("MainWindow", "Quit", 0));
        actionAbout_MrGaze->setText(QApplication::translate("MainWindow", "About MrGaze", 0));
        lineEdit->setText(QApplication::translate("MainWindow", "0", 0));
        lineEdit_2->setText(QApplication::translate("MainWindow", "100", 0));
        tabWidget->setTabText(tabWidget->indexOf(AdjustTab), QApplication::translate("MainWindow", "Adjust", 0));
        tabWidget->setTabText(tabWidget->indexOf(Calibration_Tab), QApplication::translate("MainWindow", "Calibration", 0));
        tabWidget->setTabText(tabWidget->indexOf(Tracking_Tab), QApplication::translate("MainWindow", "Tracking", 0));
        menuGaze_Estimation->setTitle(QApplication::translate("MainWindow", "File", 0));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
