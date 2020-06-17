class Employee:
    empCount=0
    sum=0
    average=0
    def __init__(self,name,family,salary,department):
        self.name=name
        self.family=family
        self.salary=salary
        self.department=department
        Employee.empCount+=1
        Employee.sum+=self.salary
        Employee.average=(Employee.sum/Employee.empCount)
    def employee_count(self):
        print("Totalemployee %d" % Employee.empCount)
    def average_sal(self):
        print("average is%d"% Employee.average)
    def display(self):
        print("name:%s" %self.name ,",family:%s" %self.family, ",salary:%d" %self.salary, ",department:%s" %self.department)
class FulltimeEmployee(Employee):
    def __init__(self,name,family,salary,department):
        print("calling another class")
    
emp1=Employee("karishma","hi",3000,"lil")
emp2=Employee("Roshini","hi",4000,"lil")
emp3=Employee("sarath","hi",5000,"lil")
emp4=Employee("pavan","hi",6000,"lil")
#print("Totalemployee %d" % Employee.empCount)
print("average is %d"% Employee.average)
print("count is %d"% FulltimeEmployee.empCount)
emp1.display()


