source : https://github.com/windowsub0406/SelfDrivingCarND/tree/master/SDC_project_1

It is a lane detection code using Raspberry pi.


## 컬렉션 처리
     
- 가변 길이 인자 : vararg키워드를 사용하면 호출시 인자 개수가 달라질 수 있는 함수를 정의할 수 있다.
    
자바 : 타입 뒤에 ...를 붙임
     
배열을 그냥 넘기면 된다.
    
    코틀린 : 파라미터 앞에 vararg 변경자를 붙임
    
    기술적으로 스프레드 연산자가 작업을 해줌, 배열 앞에 *를 붙이기만 하면 된다.
    
    #중위 함수 호출
    
    중위 함수 호출을 사용하면 인자가 하나뿐인 메소드를 간편하게 호출할 수 있다.
    
    인자가 하나뿐인 일반 메소드나 확장함수에 중위 호출을 사용할 수 있다.
    
    함수를 중위호출에 사용하게 허용하고 싶으면 infix변경자를 함수 선언 앞에 추가해야 한다.
    
    구조분해선언 - Pair 인스턴스 외 다른 객체에도 구조 분해를 적용할 수 있다.(ex 2개 변수 초기화)
    
    to함수 - 타입과 관계없이 임의의 순서쌍을 만들 수 있다.
    
    #라이브러리 지원
    
    구조 분해 선언을 사용하면 복합적인 값을 분해해서 여러 변수에 나눠 담을 수 있다.
    

## 문자열과 정규식 다루기
    
    마침표(.)는 모든 문자를 나타내는 정규식으로 해석된다.
    
    정규식을 파라미터로 받는 함수는 String이 아닌 Regex 타입의 값을 받는다.
    
    코틀린에서는 split함수에 전달하는 값의 타입의 따라 정규식이나 일반 텍스트 중 어느 것으로 문자열을 분리하는지 쉽게 알 수 있다.
    
    정규식은 강력하지만 나중에 알아보기 힘든 경우가 많다.
    
    3중 따옴표 문자열에서는 역슬래시(\)를 포함한 어떤 문자도 이스케이프할 필요가 없다.
    
    3중 따옴표 문자열 안에 $를 넣어야한다면 문자열 템플릿 안에 ‘$’문자를 넣어야한다.
    

## 코드 다듬기 : 로컬 함수와 확장
    
    코틀린에서는 함수에서 추출한 함수를 원 함수 내부에 중첩시킬 수 있다.
    
    로컬 함수를 써서 코드를 더 깔끔하게 유지하면서 중복을 제거할 수 있다.
