from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = "Krishna"
    age: Optional[int]
    email: EmailStr
    cgpa: float = Field(ge=0.0, le=10.0, default=8.5, description="Cgpa of a student in decimal value")


new_student = {'age': 22, 'email': "test@example.com"}

student = Student(**new_student)

student_dict = dict(student)

print(student_dict)
student_json = student.model_dump_json()