from django.shortcuts import render, HttpResponse
from vrp_classical import test
# Create your views here.
def home(request):
    # return HttpResponse('Hello, World!')
    test_variable = test.test_1
    return render(request, 'home.html', {'test_variable': test_variable})