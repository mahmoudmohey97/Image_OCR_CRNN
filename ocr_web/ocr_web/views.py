from django.shortcuts import redirect, render
from ocr_web import controller  

# our home page view
def home(request):
    return render(request, 'index.html')
        
# our result page view
def result(request):
    img_path = request.GET['img_name']
    result = controller.getPredictions(img_path)

    return render(request, 'result.html', {'result':result})