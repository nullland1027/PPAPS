var zero = document.getElementById('zero_div').textContent;
var one = document.getElementById('one_div').textContent;

option = {
    tooltip: {
        trigger: 'item'
    },
    legend: {
        top: '5%',
        left: 'center'
    },
    series: [
        {
            name: 'Pathogenicity Result',
            type: 'pie',
            radius: ['40%', '70%'],
            avoidLabelOverlap: false,
            itemStyle: {
                borderRadius: 10,
                borderColor: '#fff',
                borderWidth: 2
            },
            label: {
                show: false,
                position: 'center'
            },
            emphasis: {
                label: {
                    show: true,
                    fontSize: 40,
                    fontWeight: 'bold'
                }
            },
            labelLine: {
                show: false
            },
            data: [
                {value: parseInt(zero), name: 'Pathogenic'},
                {value: parseInt(one), name: 'Non-pathogenic'}
            ]
        }
    ]
};

var chart = echarts.init(document.getElementById('chart_page'));
chart.setOption(option);

var chart = echarts.init(document.getElementById('chart_page2'));
chart.setOption(option);