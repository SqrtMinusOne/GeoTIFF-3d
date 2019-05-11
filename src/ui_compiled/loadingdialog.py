# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/loadingdialog.ui'
#
# Created by: PyQt5 UI code generator 5.12.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_LoadingDialog(object):
    def setupUi(self, LoadingDialog):
        LoadingDialog.setObjectName("LoadingDialog")
        LoadingDialog.setWindowModality(QtCore.Qt.ApplicationModal)
        LoadingDialog.resize(440, 188)
        LoadingDialog.setStyleSheet("QLabel[objectName*=\"Show\"]{\n"
"    font-size: 12pt;\n"
"    font-family: Consolas;\n"
"    font-style: italic;a\n"
"}\n"
"")
        self.verticalLayout = QtWidgets.QVBoxLayout(LoadingDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label = QtWidgets.QLabel(LoadingDialog)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)
        self.operationLabel = QtWidgets.QLabel(LoadingDialog)
        self.operationLabel.setObjectName("operationLabel")
        self.gridLayout_2.addWidget(self.operationLabel, 0, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem, 0, 2, 2, 1)
        self.statusDescLabel = QtWidgets.QLabel(LoadingDialog)
        self.statusDescLabel.setObjectName("statusDescLabel")
        self.gridLayout_2.addWidget(self.statusDescLabel, 1, 0, 1, 1)
        self.statusLabel = QtWidgets.QLabel(LoadingDialog)
        self.statusLabel.setObjectName("statusLabel")
        self.gridLayout_2.addWidget(self.statusLabel, 1, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout_2)
        self.progressBar = QtWidgets.QProgressBar(LoadingDialog)
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(0)
        self.progressBar.setProperty("value", -1)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout.addWidget(self.progressBar)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_3 = QtWidgets.QLabel(LoadingDialog)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 0, 0, 1, 1)
        self.startTimeShow = QtWidgets.QLabel(LoadingDialog)
        self.startTimeShow.setObjectName("startTimeShow")
        self.gridLayout.addWidget(self.startTimeShow, 0, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 0, 2, 2, 1)
        self.label_5 = QtWidgets.QLabel(LoadingDialog)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 1, 0, 1, 1)
        self.etaShow = QtWidgets.QLabel(LoadingDialog)
        self.etaShow.setObjectName("etaShow")
        self.gridLayout.addWidget(self.etaShow, 1, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem2)
        self.logButton = QtWidgets.QPushButton(LoadingDialog)
        self.logButton.setEnabled(False)
        self.logButton.setObjectName("logButton")
        self.horizontalLayout_3.addWidget(self.logButton)
        self.cancelButton = QtWidgets.QPushButton(LoadingDialog)
        self.cancelButton.setObjectName("cancelButton")
        self.horizontalLayout_3.addWidget(self.cancelButton)
        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.retranslateUi(LoadingDialog)
        self.cancelButton.clicked.connect(LoadingDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(LoadingDialog)

    def retranslateUi(self, LoadingDialog):
        _translate = QtCore.QCoreApplication.translate
        LoadingDialog.setWindowTitle(_translate("LoadingDialog", "Подождите"))
        self.label.setText(_translate("LoadingDialog", "Идет выполнение операции:"))
        self.operationLabel.setText(_translate("LoadingDialog", "[Operation]"))
        self.statusDescLabel.setText(_translate("LoadingDialog", "Статус:"))
        self.statusLabel.setText(_translate("LoadingDialog", "[Status]"))
        self.label_3.setText(_translate("LoadingDialog", "Время начала:"))
        self.startTimeShow.setText(_translate("LoadingDialog", "startTimeShow"))
        self.label_5.setText(_translate("LoadingDialog", "Осталось:"))
        self.etaShow.setText(_translate("LoadingDialog", "Неизвестно"))
        self.logButton.setText(_translate("LoadingDialog", "Открыть лог"))
        self.cancelButton.setText(_translate("LoadingDialog", "Отмена"))


