## Even or Odd
```
%{ 
#include<stdio.h> 
int i; 
%} 

%% 

[0-9]+	 {i=atoi(yytext); 
		if(i%2==0) 
			printf("Even"); 
		else
		printf("Odd");} 
%% 

int yywrap(){} 

/* Driver code */
int main() 
{ 

	yylex(); 
	return 0; 
}
// lex evenorodd.l
// gcc lex.yy.c
// ./a.out
```
## Count no of vowels and consonants
```
%{
	int vow_count=0;
	int const_count =0;
%}

%%
[aeiouAEIOU] {vow_count++;}
[a-zA-Z] {const_count++;}
%%
int yywrap(){}
int main()
{
	printf("Enter the string of vowels and consonants:");
	yylex();
	printf("Number of vowels are: %d\n", vow_count);
	printf("Number of consonants are: %d\n", const_count);
	return 0;
} 
// lex vowel.l
// cc lex.yy.c -lfl
// ./a.out
```
## Count no of digits
```
%{
#include <stdio.h>
#include <string.h>
int digit_count = 0;
%}

%%
[0-9]+ {
    digit_count++;
    printf("Integer: %s, Number of digits: %d\n", yytext, (int)strlen(yytext));
}
%%

int main() {
    yylex();
    return 0;
}
// lex digits.l
// gcc lex.yy.c
// ./a.out
```
## No of words, characters, lines, spaces
```
%{
#include<stdio.h>
int lc=0,sc=0,tc=0,ch=0,wc=0;        // GLOBAL VARIABLES
%}
 
// RULE SECTION
%%
[\n] { lc++; ch+=yyleng;}
[  \t] { sc++; ch+=yyleng;}
[^\t] { tc++; ch+=yyleng;}
[^\t\n ]+ { wc++;  ch+=yyleng;}  
%%
 
int yywrap(){ return 1;    }
/*        After inputting press ctrl+d         */
 
// MAIN FUNCTION
int main(){
    printf("Enter the Sentence : ");
    yylex();
    printf("Number of lines : %d\n",lc);
    printf("Number of spaces : %d\n",sc);
    printf("Number of tabs, words, charc : %d , %d , %d\n",tc,wc,ch);
     
    return 0;
}
// lex filename.l
// gcc lex.yy.c
// ./a.out
```
## Lex - Calculator
```
%{
#include <stdio.h>
#include <stdlib.h>
#undef yywrap
#define yywrap() 1 
int f1 = 0, f2 = 0;
char oper;
float op1 = 0, op2 = 0, ans = 0;
void eval();
%}
DIGIT [0-9]
NUM {DIGIT}+(\.{DIGIT}+)?
OP [*/+-]
%%
{NUM} {
    if (f1 == 0) {
        op1 = atof(yytext);
        f1 = 1;
    } 
    else if (f2 == -1) {
        op2 = atof(yytext);
        f2 = 1;
    }
    if ((f1 == 1) && (f2 == 1)) {
        eval();
        f1 = 0;
        f2 = 0;
    }
}
{OP} {
    oper = yytext[0];
    f2 = -1;
}
[\n] {
    if (f1 == 1 && f2 == 1) {
        eval();
        f1 = 0;
        f2 = 0;
    }
}	
%%
int main() {
    printf("Enter an arithmetic expression:\n");
    yylex();
    return 0;
}
void eval() {
    switch (oper) {
        case '+':
            ans = op1 + op2;
            break;
        case '-':
            ans = op1 - op2;
            break;
        case '*':
            ans = op1 * op2;
            break;
        case '/':
            if (op2 == 0) {
                printf("ERROR: Division by zero\n");
                return;
            } else {
                ans = op1 / op2;
            }
            break;
        default:
            printf("ERROR: Invalid operator\n");
            return;
    }
    printf("The answer is: %f\n", ans);
}
// lex cal.c
// gcc lex.yy.c
// ./a.out
```
## Exp - 4: Lexical Analyzer
```
//lexp.l
%{
int COMMENT=0;
%}
identifier [a-zA-Z][a-zA-Z0-9]*
%%
#.* {printf ("\n %s is a Preprocessor Directive",yytext);}
int |
float |
main |
if |
else |
printf |
scanf |
for |
char |
getch |
while {printf("\n %s is a Keyword",yytext);}
"/*" {COMMENT=1;}
"*/" {COMMENT=0;}
{identifier}\( {if(!COMMENT) printf("\n Function:\t %s",yytext);}
\{ {if(!COMMENT) printf("\n Block Begins");
\} {if(!COMMENT) printf("\n Block Ends");}
{identifier}(\[[0-9]*\])? {if(!COMMENT) printf("\n %s is an Identifier",yytext);}
\".*\" {if(!COMMENT) printf("\n %s is a String",yytext);}
[0-9]+ {if(!COMMENT) printf("\n %s is a Number",yytext);}
\)(\;)? {if(!COMMENT) printf("\t");ECHO;printf("\n");}
\( ECHO;
= {if(!COMMENT) printf("\n%s is an Assmt oprtr",yytext);}
\<= |
\>= |
\< |
== {if(!COMMENT) printf("\n %s is a Rel. Operator",yytext);}
.|\n
%%
int main(int argc, char **argv)
{
if(argc>1)
{
FILE *file;
file=fopen(argv[1],"r");
if(!file)
{
printf("\n Could not open the file: %s",argv[1]);
exit(0);
}
yyin=file;
}
yylex();
printf("\n\n");
return 0;
}
int yywrap()
{
return 0;
}
//Output:
//test.c
#include<stdio.h>
main()
{
int fact=1,n;
for(int i=1;i<=n;i++)
{ fact=fact*i; }
printf("Factorial Value of N is", fact);
getch();
}
//$ lex lexp.l
//$ cc lex.yy.c
//$ ./a.out test.c
```
## Lexical, Syntax and Semantic Errors
### Lexical Error
```
#include <iostream>
using namespace std;

int main() {
    int num = 10;
    int @value = 20; 
    // Lexical error: '@' is not a valid character in identifier names

    cout << num + @value << endl;
    return 0;
}
```
### Syntax Error
```
#include <iostream>
using namespace std;

int main() {
    int x = 10
    cout << "Value of x is: " << x << endl;
    // Missing semicolon ; after int x = 10 leads to a syntax error
    return 0;
}
```
### Semantic Error
```
#include <iostream>
using namespace std;

int main() {
    int totalApples = 10;
    int totalOranges = 7;

    int totalFruits = totalApples - totalOranges; 
    // Semantic error: Wrong logic (should be addition)
    cout << "No. of Apples: " << totalApples << endl;
    cout << "No. of Oranges: " << totalOranges << endl;
    cout << "Total fruits: " << totalFruits << endl;
    return 0;
}
```
## SR
```
#include <iostream>
#include <cstring>
using namespace std;

char input[16] = "32423";     // Input string
char stack[20];               // Parsing stack
int i = 0, j = 0, top = 0;    // Indices for parsing

// Function to perform reductions
void reduce() {
    const char* rule = "REDUCE TO E -> ";

    // Rule: E -> 4
    for (int k = 0; k < top; k++) {
        if (stack[k] == '4') {
            cout << rule << "4\n";
            stack[k] = 'E';
            stack[k + 1] = '\0';
            top = k + 1;
            cout << "$" << stack << "\t" << input << "$\t";
        }
    }

    // Rule: E -> 2 E 2
    for (int k = 0; k < top - 2; k++) {
        if (stack[k] == '2' && stack[k + 1] == 'E' && stack[k + 2] == '2') {
            cout << rule << "2E2\n";
            stack[k] = 'E';
            top = k + 1;
            stack[top] = '\0';
            cout << "$" << stack << "\t" << input << "$\t";
        }
    }

    // Rule: E -> 3 E 3
    for (int k = 0; k < top - 2; k++) {
        if (stack[k] == '3' && stack[k + 1] == 'E' && stack[k + 2] == '3') {
            cout << rule << "3E3\n";
            stack[k] = 'E';
            top = k + 1;
            stack[top] = '\0';
            cout << "$" << stack << "\t" << input << "$\t";
        }
    }
}

int main() {
    int len = strlen(input);

    // Show productions first
    cout << "GRAMMAR:\n";
    cout << "E -> 4\n";
    cout << "E -> 2E2\n";
    cout << "E -> 3E3\n\n";

    // Show header for parsing trace
    cout << "Stack\tInput\tAction\n";
    cout << "$\t" << input << "$\t\n";

    while (j < len) {
        cout << "SHIFT\n";
        stack[top++] = input[j];
        stack[top] = '\0';
        input[j++] = ' ';
        cout << "$" << stack << "\t" << input << "$\t";
        reduce();
    }

    reduce();  // Final check

    if (stack[0] == 'E' && stack[1] == '\0')
        cout << "\nAccept\n";
    else
        cout << "\nReject\n";

    return 0;
}
```
## OPG
```
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX 100

// Operator Precedence Table for + - * / $
char precedence_table[5][5] = {
    //   +    -    *    /    $
    {'>', '>', '<', '<', '>'}, // +
    {'>', '>', '<', '<', '>'}, // -
    {'>', '>', '>', '>', '>'}, // *
    {'>', '>', '>', '>', '>'}, // /
    {'<', '<', '<', '<', 'A'}  // $
};

char operators[] = "+-*/$";

int getIndex(char symbol) {
    for (int i = 0; i < 5; i++)
        if (operators[i] == symbol)
            return i;
    return -1;
}

// Stack structure
typedef struct {
    char data[MAX];
    int top;
} Stack;

void push(Stack *s, char ch) {
    s->data[++(s->top)] = ch;
}

char pop(Stack *s) {
    return s->data[(s->top)--];
}

char peek(Stack *s) {
    return s->data[s->top];
}

int isOperator(char ch) {
    return strchr(operators, ch) != NULL;
}

void parseExpression(char *input) {
    Stack stack = {.top = -1};
    push(&stack, '$');

    int i = 0;
    printf("Stack\tInput\tAction\n");

    while (1) {
        while (input[i] == ' ') i++;

        // Print stack
        for (int j = 0; j <= stack.top; j++)
            printf("%c", stack.data[j]);
        printf("\t%s\t", &input[i]);

        char top = peek(&stack);
        char next = input[i];

        if (isalnum(next)) {
            printf("Shift (Operand: %c)\n", next);
            i++;
            continue;
        }

        int ti = getIndex(top), ni = getIndex(next);
        if (ti == -1 || ni == -1) {
            printf("Error: Invalid character\n");
            return;
        }

        char prec = precedence_table[ti][ni];
        if (prec == '<' || prec == '=') {
            push(&stack, next);
            printf("Shift\n");
            i++;
        } else if (prec == '>') {
            pop(&stack);
            printf("Reduce\n");
        } else if (prec == 'A' && top == '$' && next == '$') {
            printf("Input accepted!\n");
            return;
        } else {
            printf("Error: Invalid precedence\n");
            return;
        }
    }
}

int main() {
    char input[MAX];
    printf("Enter expression ending with $: ");
    fgets(input, MAX, stdin);
    parseExpression(input);
    return 0;
}
```
## LR(0)
```
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <stack>
#include <algorithm>
using namespace std;

struct Item {
    string lhs, rhs;
    int dot;
    bool operator==(const Item& other) const {
        return lhs == other.lhs && rhs == other.rhs && dot == other.dot;
    }
};

struct State {
    vector<Item> items;
    map<char, int> transitions;
};

vector<string> productions;
vector<State> states;
map<pair<int, char>, string> ACTION;
map<pair<int, char>, int> GOTO;
set<char> terminals, nonTerminals;

bool hasItem(const vector<Item>& items, const Item& item) {
    return find(items.begin(), items.end(), item) != items.end();
}

vector<Item> closure(vector<Item> items) {
    bool changed;
    do {
        changed = false;
        vector<Item> newItems = items;
        for (auto& item : items) {
            if (item.dot < item.rhs.size()) {
                char symbol = item.rhs[item.dot];
                if (nonTerminals.count(symbol)) {
                    for (auto& prod : productions) {
                        if (prod[0] == symbol) {
                            Item newItem = {prod.substr(0, 1), prod.substr(3), 0};
                            if (!hasItem(newItems, newItem)) {
                                newItems.push_back(newItem);
                                changed = true;
                            }
                        }
                    }
                }
            }
        }
        items = newItems;
    } while (changed);
    return items;
}

vector<Item> goTo(const vector<Item>& items, char symbol) {
    vector<Item> moved;
    for (auto& item : items) {
        if (item.dot < item.rhs.size() && item.rhs[item.dot] == symbol) {
            moved.push_back({item.lhs, item.rhs, item.dot + 1});
        }
    }
    return closure(moved);
}

int findState(const vector<Item>& items) {
    for (int i = 0; i < states.size(); i++) {
        if (states[i].items == items) return i;
    }
    return -1;
}

void constructAutomaton() {
    productions.insert(productions.begin(), "S'->S");
    states.push_back({closure({{"S'", "S", 0}})});

    for (int i = 0; i < states.size(); i++) {
        for (char symbol : terminals) {
            vector<Item> next = goTo(states[i].items, symbol);
            if (!next.empty()) {
                int idx = findState(next);
                if (idx == -1) {
                    states.push_back({next});
                    idx = states.size() - 1;
                }
                states[i].transitions[symbol] = idx;
            }
        }
        for (char symbol : nonTerminals) {
            vector<Item> next = goTo(states[i].items, symbol);
            if (!next.empty()) {
                int idx = findState(next);
                if (idx == -1) {
                    states.push_back({next});
                    idx = states.size() - 1;
                }
                states[i].transitions[symbol] = idx;
            }
        }
    }

    for (int i = 0; i < states.size(); i++) {
        for (auto& item : states[i].items) {
            if (item.dot < item.rhs.size()) {
                char symbol = item.rhs[item.dot];
                if (terminals.count(symbol)) {
                    ACTION[{i, symbol}] = "S" + to_string(states[i].transitions[symbol]);
                } else {
                    GOTO[{i, symbol}] = states[i].transitions[symbol];
                }
            } else {
                string rule = item.lhs + "->" + item.rhs;
                if (item.lhs == "S'") {
                    ACTION[{i, '$'}] = "ACCEPT";
                } else {
                    for (char t : terminals) {
                        ACTION[{i, t}] = "R(" + rule + ")";
                    }
                    ACTION[{i, '$'}] = "R(" + rule + ")";
                }
            }
        }
    }
}

void printTable() {
    cout << "\nLR(0) Parsing Table:\n";
    cout << "State\t";
    for (char t : terminals) cout << t << "\t";
    cout << "$\t|\t";
    for (char nt : nonTerminals) cout << nt << "\t";
    cout << endl;

    for (int i = 0; i < states.size(); i++) {
        cout << i << "\t";
        for (char t : terminals)
            cout << (ACTION.count({i, t}) ? ACTION[{i, t}] : "-") << "\t";
        cout << (ACTION.count({i, '$'}) ? ACTION[{i, '$'}] : "-") << "\t|\t";
        for (char nt : nonTerminals)
            cout << (GOTO.count({i, nt}) ? to_string(GOTO[{i, nt}]) : "-") << "\t";
        cout << endl;
    }
}

bool parse(string input) {
    stack<int> s;
    s.push(0);
    input += "$";
    int i = 0;

    while (true) {
        int state = s.top();
        char sym = input[i];

        if (ACTION.count({state, sym})) {
            string action = ACTION[{state, sym}];

            if (action[0] == 'S') {
                s.push(stoi(action.substr(1)));
                i++;
            } else if (action[0] == 'R') {
                string rule = action.substr(2, action.size() - 3);
                int pos = rule.find("->");
                string rhs = rule.substr(pos + 2);
                for (int j = 0; j < rhs.size(); j++) s.pop();
                int prev = s.top();
                s.push(GOTO[{prev, rule[0]}]);
            } else if (action == "ACCEPT") {
                return true;
            }
        } else {
            return false;
        }
    }
}

int main() {
    int n;
    cout << "Enter number of productions: ";
    cin >> n;
    cout << "Enter productions (Format: A->BC or A->ε):\n";
    for (int i = 0; i < n; i++) {
        string p;
        cin >> p;
        productions.push_back(p);
        nonTerminals.insert(p[0]);
        for (char c : p.substr(3)) {
            if (!isupper(c) && c != 'ε') terminals.insert(c);
        }
    }

    constructAutomaton();
    printTable();

    string str;
    cout << "\nEnter input string: ";
    cin >> str;

    if (parse(str)) cout << "String accepted!\n";
    else cout << "String rejected!\n";
    return 0;
}
```
## Three-Address Code
```
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

int tempVar = 1; // Counter for temporary variables

// Function to generate three-address code
void generateTAC(char op, char arg1[], char arg2[], char result[]) {
    printf("%s = %s %c %s\n", result, arg1, op, arg2);
}

// Function to check if a character is an operator
int isOperator(char c) {
    return (c == '+' || c == '-' || c == '*' || c == '/');
}

// Function to generate TAC for simple expressions
void processExpression(char expr[]) {
    char tokens[10][10]; // Store tokens
    int tokenCount = 0;
    
    // Tokenizing the input expression
    char *token = strtok(expr, " ");
    while (token != NULL) {
        strcpy(tokens[tokenCount++], token);
        token = strtok(NULL, " ");
    }

    if (tokenCount < 3) {
        printf("Invalid expression format!\n");
        return;
    }

    char temp1[10], temp2[10], result[10];

    // Handling operator precedence (* and / first)
    for (int i = 1; i < tokenCount; i += 2) {
        if (tokens[i][0] == '*' || tokens[i][0] == '/') {
            sprintf(result, "t%d", tempVar++);
            generateTAC(tokens[i][0], tokens[i - 1], tokens[i + 1], result);
            strcpy(tokens[i - 1], result);
            for (int j = i; j < tokenCount - 2; j++) {
                strcpy(tokens[j], tokens[j + 2]);
            }
            tokenCount -= 2;
            i -= 2; // Re-evaluate after shifting tokens
        }
    }

    // Handling + and -
    for (int i = 1; i < tokenCount; i += 2) {
        if (tokens[i][0] == '+' || tokens[i][0] == '-') {
            sprintf(result, "t%d", tempVar++);
            generateTAC(tokens[i][0], tokens[i - 1], tokens[i + 1], result);
            strcpy(tokens[i - 1], result);
            for (int j = i; j < tokenCount - 2; j++) {
                strcpy(tokens[j], tokens[j + 2]);
            }
            tokenCount -= 2;
            i -= 2; // Re-evaluate after shifting tokens
        }
    }
}

int main() {
    char expression[50];

    // Taking user input
    printf("Enter an arithmetic expression (e.g., a + b * c - d): ");
    fgets(expression, sizeof(expression), stdin);
    expression[strcspn(expression, "\n")] = '\0'; // Remove newline

    processExpression(expression);

    return 0;
}
```
## Symbol Table
```
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

#define MAX 100

int main() {
    char expr[MAX], ch;
    void *addr[50];
    char symbol[50];
    int i = 0, x = 0;

    printf("Enter an expression terminated by $: ");
    while ((ch = getchar()) != '$' && i < MAX - 1) {
        expr[i++] = ch;
    }
    expr[i] = '\0';

    printf("\nGiven Expression: %s", expr);

    printf("\n\nSymbol Table");
    printf("\nSymbol\tAddress\t\tType");

    for (i = 0; expr[i] != '\0'; i++) {
        ch = expr[i];

        if (isalpha(ch)) {
            // Identifier
            addr[x] = malloc(sizeof(char));
            symbol[x] = ch;
            printf("\n%c\t%p\tidentifier", ch, addr[x]);
            x++;
        } else if (ch == '+' || ch == '-' || ch == '*' || ch == '/' || ch == '=') {
            // Operator
            addr[x] = malloc(sizeof(char));
            symbol[x] = ch;
            printf("\n%c\t%p\toperator", ch, addr[x]);
            x++;
        }
    }

    // Free allocated memory
    for (i = 0; i < x; i++) {
        free(addr[i]);
    }

    return 0;
}
```
## Intermediate Code Generation
```
#include <iostream>
#include <vector>
#include <sstream>
using namespace std;

// Function to generate Assembly from Three-Address Code (TAC)
void generateAssembly(const vector<string>& tac) {
    cout << "\nGenerated Assembly Code:\n";

    for (const string& instruction : tac) {
        stringstream ss(instruction);
        string result, equalSign, op1, op, op2;
        ss >> result >> equalSign >> op1;  // Extract first operand

        if (!(ss >> op >> op2)) {
            // Simple assignment (t1 = a)
            cout << "MOV " << result << ", " << op1 << endl;
        } else {
            // Arithmetic operation (t1 = b * c)
            cout << "MOV R1, " << op1 << endl;
            if (op == "+") cout << "ADD R1, " << op2 << endl;
            else if (op == "-") cout << "SUB R1, " << op2 << endl;
            else if (op == "*") cout << "MUL R1, " << op2 << endl;
            else if (op == "/") cout << "DIV R1, " << op2 << endl;
            cout << "MOV " << result << ", R1" << endl;
        }
    }
}

int main() {
    vector<string> tac;
    int n;

    cout << "Enter the number of Three-Address Code (TAC) instructions: ";
    cin >> n;
    cin.ignore();  // Ignore newline

    cout << "Enter TAC instructions (Format: tX = operand1 op operand2):\n";
    for (int i = 0; i < n; i++) {
        string instruction;
        getline(cin, instruction);
        tac.push_back(instruction);
    }

    generateAssembly(tac);
    return 0;
}
```
## TAC + ICG
```
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX 100

//------------------------------------------------[2]: Global Declarations

char expr[MAX];              // Input expression
int tempVarCount = 1;        // Temporary variable counter (t1, t2, ...)

char op[MAX];                // Operator stack
int opTop = -1;

char operand[MAX][10];       // Operand stack (holds variables or temp results)
int operandTop = -1;

typedef struct {
    char result[10];         // Temporary result variable (t1, t2...)
    char operand1[10];       // Left-hand side operand
    char operand2[10];       // Right-hand side operand
    char op;                 // Operator (+, -, *, /)
} TAC;

TAC tacList[MAX];            // Array to hold all TAC instructions
int tacIndex = 0;            // Current number of TAC instructions

int precedence(char);
void generateTAC();
void generateAssembly();
void pushOperator(char);
char popOperator();
void pushOperand(char *);
char *popOperand();

int main() {
    printf("Enter an arithmetic expression: ");
    scanf("%s", expr);

    printf("\nThree Address Code (TAC):\n");
    generateTAC();

    printf("\nEquivalent Assembly Code:\n");
    generateAssembly();

    return 0;
}

int precedence(char op) {
    if (op == '+' || op == '-') return 1;
    if (op == '*' || op == '/') return 2;
    return 0;
}

void generateTAC() {
    char temp[10];
    int i, j = 0, length = strlen(expr);

    for (i = 0; i < length; i++) {
        if (isalnum(expr[i])) {
            // Build full operand (e.g., a, b, x1)
            temp[j++] = expr[i];
            if (i == length - 1 || !isalnum(expr[i + 1])) {
                temp[j] = '\0';
                pushOperand(temp);
                j = 0;
            }
        } 
        else if (expr[i] == '(') {
            pushOperator(expr[i]);
        } 
        else if (expr[i] == ')') {
            while (opTop >= 0 && op[opTop] != '(') {
                char oper = popOperator();
                char *right = popOperand();
                char *left = popOperand();

                sprintf(tacList[tacIndex].result, "t%d", tempVarCount);
                strcpy(tacList[tacIndex].operand1, left);
                strcpy(tacList[tacIndex].operand2, right);
                tacList[tacIndex].op = oper;
                tacIndex++;

                char tempVarStr[10];
                sprintf(tempVarStr, "t%d", tempVarCount++);
                pushOperand(tempVarStr);
            }
            popOperator(); // Pop '('
        } 
        else {
            while (opTop >= 0 && precedence(op[opTop]) >= precedence(expr[i])) {
                char oper = popOperator();
                char *right = popOperand();
                char *left = popOperand();

                sprintf(tacList[tacIndex].result, "t%d", tempVarCount);
                strcpy(tacList[tacIndex].operand1, left);
                strcpy(tacList[tacIndex].operand2, right);
                tacList[tacIndex].op = oper;
                tacIndex++;

                char tempVarStr[10];
                sprintf(tempVarStr, "t%d", tempVarCount++);
                pushOperand(tempVarStr);
            }
            pushOperator(expr[i]);
        }
    }

    // Remaining operators
    while (opTop >= 0) {
        char oper = popOperator();
        char *right = popOperand();
        char *left = popOperand();

        sprintf(tacList[tacIndex].result, "t%d", tempVarCount);
        strcpy(tacList[tacIndex].operand1, left);
        strcpy(tacList[tacIndex].operand2, right);
        tacList[tacIndex].op = oper;
        tacIndex++;

        char tempVarStr[10];
        sprintf(tempVarStr, "t%d", tempVarCount++);
        pushOperand(tempVarStr);
    }

    // Print TAC
    for (i = 0; i < tacIndex; i++) {
        printf("  %s := %s %c %s\n", tacList[i].result, tacList[i].operand1, tacList[i].op, tacList[i].operand2);
    }
}

void generateAssembly() {
    for (int i = 0; i < tacIndex; i++) {
        printf("  MOV R0, %s\n", tacList[i].operand1);
        switch (tacList[i].op) {
            case '+': printf("  ADD R0, %s\n", tacList[i].operand2); break;
            case '-': printf("  SUB R0, %s\n", tacList[i].operand2); break;
            case '*': printf("  MUL R0, %s\n", tacList[i].operand2); break;
            case '/': printf("  DIV R0, %s\n", tacList[i].operand2); break;
        }
        printf("  MOV %s, R0\n", tacList[i].result);
    }
}
void pushOperator(char opr) {
    op[++opTop] = opr;
}
char popOperator() {
    return op[opTop--];
}
void pushOperand(char *val) {
    strcpy(operand[++operandTop], val);
}
char *popOperand() {
    return operand[operandTop--];
}
```
