students=[]

def add_student():
    name=input("Enter a name:")
    roll_no=input(input("Enter a roll number: "))
    age=int(input("Enter your age:"))
    gender=input("Enter Gender:")

    student={'name':name,'roll_no':roll_no,'age':age,'gender':gender}
    students.append(student)
    print("Student added data successfully\n")

def view_student():
    if len(students)==0:
        print("No Students Data Is Found")
        return
    print("Students Data")

    for i,student in enumerate(students, start=1):
        print(f"{i}name:{student['name']}, roll_no:{student['roll_no']}, age:{student['age']}, gender:{student['gender']}")

    print()

def search_student():
    roll_no =input("search by roll number:")
    data_found=False
    for student in students:
        if student['roll_no'] == roll_no:
            data_found=True
            print(f"Name:{student['name']},RollNo:{student[roll_no]},")
            break
        if data_found==False:
                print('No data')

def remove_student():
    roll_no=input("Enter a roll no:")
    global students
    students=[student for student in students if student['roll_no']!= roll_no]
    print('Student Data Removed\n')

def main():
    while True:
    print("Enter 1 to Add Student:")
    print("Enter 2 to View Student:")
    print("Enter 3 to Search Student:")
    print("Enter 4 to Remove Student:")

    ch=int(input("Enter a Choice:"))

    if ch==1:
        def add_student()
    elif:



search_student()
view_student()
add_student()
add_student()
view_student()
search_student()
remove_student()
