# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'modifystudent_password.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
import studentinfo
import sys
import sqlite3
import updatingdb

class Ui_modifystudent_password(object):
    def setupUi(self, modifystudent_password):
        modifystudent_password.setObjectName("modifystudent_password")
        modifystudent_password.resize(340, 90)
        self.centralwidget = QtWidgets.QWidget(modifystudent_password)
        self.centralwidget.setObjectName("centralwidget")
        self.password_change = QtWidgets.QLineEdit(self.centralwidget)
        self.password_change.setGeometry(QtCore.QRect(130, 20, 191, 22))
        self.password_change.setObjectName("password_change")
        self.cancel_button = QtWidgets.QPushButton(self.centralwidget)
        self.cancel_button.setGeometry(QtCore.QRect(230, 50, 93, 28))
        self.cancel_button.setObjectName("cancel_button")
        self.okay_button = QtWidgets.QPushButton(self.centralwidget)
        self.okay_button.setGeometry(QtCore.QRect(130, 50, 93, 28))
        self.okay_button.setObjectName("okay_button")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 20, 121, 16))
        self.label.setObjectName("label")
        modifystudent_password.setCentralWidget(self.centralwidget)

        self.retranslateUi(modifystudent_password)
        QtCore.QMetaObject.connectSlotsByName(modifystudent_password)
        ################################################################
        self.cancel_button.clicked.connect(modifystudent_password.close)
        self.okay_button.clicked.connect(self.showupdatingdb)
        self.okay_button.clicked.connect(modifystudent_password.close)

    def showupdatingdb(self):
        self.updatingdbWindow = QtWidgets.QMainWindow()
        self.ui = updatingdb.Ui_updatedb()
        self.ui.setupUi(self.updatingdbWindow)
        self.updatingdbWindow.show() 
 
        password = self.password_change.text()
        try:
            con = sqlite3.connect("userlist.db",timeout=10)
            cur = con.cursor()
            cur.execute('SELECT * from users WHERE ID =2')
            one = cur.fetchall()
            con.commit()
            for row in one:
                modify = row[5]
            print(modify)
            print(password)

            cur.execute('UPDATE users SET PASSWORD = ? WHERE ID= 2',(password,))
            con.commit()
            cur.execute('UPDATE users SET PASSWORD = ? WHERE ID= ?',(password,modify))
            data = cur.fetchall()
            con.commit()
            for row in data:
                password = row[4]
            con.close()
            self.label.setText(password)
        except Exception:
            self.showMessageBox('Database Error','Could not access the database')

    def showMessageBox(self,title,message):
        msgBox = QtWidgets.QMessageBox()
        msgBox.setIcon(QtWidgets.QMessageBox.Warning)
        msgBox.setWindowTitle(title)
        msgBox.setText(message)
        msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msgBox.exec_()

        
    def retranslateUi(self, modifystudent_password):
        _translate = QtCore.QCoreApplication.translate
        modifystudent_password.setWindowTitle(_translate("modifystudent_password", "Modify Student"))
        self.cancel_button.setText(_translate("modifystudent_password", "Cancel"))
        self.okay_button.setText(_translate("modifystudent_password", "Ok"))
        self.label.setText(_translate("modifystudent_password", "New password:"))

if __name__ == "__main__":
        app = QtWidgets.QApplication(sys.argv)
        modifyinfo = QtWidgets.QMainWindow()
        ui = Ui_modifystudent_password()
        ui.setupUi(modifyinfo)
        modifyinfo.show()
        sys.exit(app.exec_())