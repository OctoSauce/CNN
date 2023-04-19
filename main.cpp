#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <time.h>
#include <math.h>
#include <fstream>
#include <sstream>
#include <random>

#define lldd long double
#define rep(i,n) for(ll i =0;i<n;i++)
#define ll long long
#define esp 0.00000000000001



using namespace std;


default_random_engine engine;
uniform_real_distribution<lldd> gen(0.0,1.0);
struct mat{   // Structure to hold matrix data of images
    ll n_dim;
    lldd ***input = nullptr;
    ll input_size;
    mat(ll size,ll ndim)
    {
        n_dim = ndim;
        input_size = size;
        input = new lldd**[n_dim];
        rep(i,n_dim)
        {
            input[i] = new lldd*[input_size];
            rep(j,input_size)
            {
                input[i][j] = new lldd[input_size];
            }
        }
//        rep(k,n_dim)
//        {
//            rep(i,input_size)
//            {
//                rep(j,input_size)
//                {
//                    input[k][i][j] = in[k][i][j];
//                }
//            }
//        }
        
    }
    //DeepCopy function
//    void make(lldd ***in)
//    {
//        rep(k,n_dim)
//        {
//            rep(i,input_size)
//            {
//                rep(j,input_size)
//                {
//                    input[k][i][j] = in[k][i][j];
//                }
//            }
//        }
//    }
    void copy(mat*inp)
    {
        input_size = inp->input_size;
        n_dim = inp->n_dim;
        rep(k,n_dim)
        {
            rep(i,input_size)
            {
                rep(j,input_size)
                {
                    input[k][i][j] = inp->input[k][i][j];
                }
            }
        }
    }
    void del()
    {
        rep(i,input_size)
        {
            rep(j,input_size)
            {
                if(input[i][j]!=nullptr)
                {
                    delete[] input[i][j];
                }
            }
                   if(input[i]!=nullptr)
                   {
                       delete[] input[i];
                   }
            
        }
        if(input!=nullptr)
        delete[] input;
    }
};

struct flat{   // Structure to hold flattened data for dense layer
    lldd *in;
    ll in_size;
    flat(ll size)
    {
        in_size = size;
        in = new lldd[in_size];
//        rep(i,in_size)
//        {
//            in[i] = inp[i];
//        }
    }
//    void make(lldd *inp,ll size)
//    {
//        rep(i,size)
//        {
//            in[i] = inp[i];
//        }
//        in_size = size;
//    }
    void copy(flat*inp)
    {
        rep(i,in_size)
        {
            in[i] = inp->in[i];
        }
//        in_size = inp->in_size;
    }
    void del()
    {
        delete[] in;
    }
};

flat* softmax(flat* inp) //Takes dense layer and returns prob values
{
    lldd maxi = 0;
    flat *out = new flat(inp->in_size);
    out->copy(inp);
    rep(i,inp->in_size)
    {
        maxi += exp(inp->in[i]);
    }
    rep(i,inp->in_size)
    {
        out->in[i] = (exp(inp->in[i])/maxi);
    }
    return out;
}

lldd sigmoid(lldd ans)
{
    return (1/(1+exp(-ans)));
}

//long double diffsigmoid(long double ans)
//{
//    long double fin = sigmoid(ans);
//    return fin*(1-fin);
//}

flat *flatten(mat* inp){ //Flattens the feature matrix to a dense layer
    flat *ans = new flat((inp->input_size)*(inp->input_size)*(inp->n_dim));
    rep(k,inp->n_dim)
    {
        rep(i,inp->input_size)
        {
            rep(j,inp->input_size)
            {
                ans->in[k*(inp->input_size)*(inp->input_size)+(i*(inp->input_size))+j] = inp->input[k][i][j];
                // cout << out[(i*inp->input_size)+j] << "\n";
            }
        }
    }
//    cout << "Input is being Flattened\n";
    return ans;
}



//Conv Layer
class conv{
    ll n_ker;
    // bool pad;
    // ll padding;
    ll ker_size;
    ll stride;
    ll input_dim;
    lldd ***kernel;
    ll output_dim;
    mat *err_rate;
    mat *in_err_rate;
    mat *next_err_rate;
    mat *del_wt;
//    lldd **output;
    ll mul;
    mat *in;
    mat *out;
public:
    conv(){
        in_err_rate = nullptr;
        printf("Enter Number of Kernels\n");
        scanf(" %lld",&n_ker);
        printf("Enter dimension of kernel\n");
        scanf(" %lld",&ker_size);
        // printf("Do you want padding (1-True,0-False)\n");
        // scanf("%d",&pad);
        // if(pad){
        //     printf("Enter size of padding\n");
        //     scanf("%d",&padding);
        // }
        printf("Enter Stride\n");
        scanf(" %lld",&stride);
        del_wt=nullptr;
        kernel  = new lldd**[n_ker];
        rep(i,n_ker)
        {
            kernel[i] = new lldd*[ker_size];
            rep(j,ker_size)
            {
                kernel[i][j] = new lldd[ker_size];
            }
        }
        
        srand((unsigned int)time(0));
        rep(k,n_ker)
        {
            rep(i,ker_size)
            {
                rep(j,ker_size)
                {
                    if(rand()%2==0)
                    {
                        mul = 1;
                    }
                    else
                    {
                        mul = -1;
                    }
                    kernel[k][i][j] = (lldd)mul*(lldd)gen(engine);
//                    kernel[k][i][j] = (lldd)mul*(lldd)rand()/(lldd)5897;
                    // cout << kernel[i][j] << " ";
                }
                // cout << "\n";
            }
        }
        in = nullptr;
        out = nullptr;
//        output = nullptr;
        // cout << "\n\n\n";
    }
    void save(ofstream&outdata)
    {
        outdata << n_ker << "," << ker_size <<"\n";
        rep(i,n_ker)
        {
            rep(j,ker_size)
            {
                rep(k,ker_size)
                {
                    if(k!=ker_size-1)
                    {
                        outdata << kernel[i][j][k] << ",";
                    }
                    else
                    {
                        outdata << kernel[i][j][k] << "\n";
                    }
                }
            }
        }
    }
    mat* deflatten(flat* inp)
    {
        if(err_rate==nullptr){
            err_rate = new mat(output_dim,n_ker);
        }
        rep(k,n_ker)
        {
            rep(i,output_dim)
            {
                rep(j,output_dim)
                {
                     err_rate->input[k][i][j] = inp->in[k*(output_dim)*(output_dim)+(i*(output_dim))+j];
                    // cout << out[(i*inp->input_size)+j] << "\n";
                }
            }
        }
        return err_rate;
    }
//    ~conv(){
//        try {
//            delete(kernel);
//        } catch (...) {
//            // Do Nothing
//        }
//        try {
//            delete(in);
//        } catch (...) {
//            // Do Nothing
//        }
//        try {
//            delete(out);
//        } catch (...) {
//            // Do Nothing
//        }
//    }

    mat *conv_op(mat *inp) //Convolution operation in a CNN convolution layer
    {
        if(in==nullptr)
        {
            in = new mat(inp->input_size,1);
        }
        rep(i,inp->input_size)
        {
            rep(j,inp->input_size)
            {
                in->input[0][i][j] = 0.0;
                rep(k,inp->n_dim)
                {
                    in->input[0][i][j]+=inp->input[k][i][j];
                }
            }
        }
        input_dim = inp->n_dim;
//        in->copy(inp);
        // output_dim = ((inp->input_size+ 2*pad - ker_size)/stride) + 1;
        output_dim = ((inp->input_size - ker_size)/stride) + 1;
        if(out==nullptr)
        {
            out = new mat(output_dim,n_ker);
        }
//        if(output==nullptr)
//        {
//            output= new lldd*[output_dim];
//            rep(i,output_dim)
//            {
//                output[i] = new lldd[output_dim];
//            }
//        }
        
        for(ll i =0;i+ker_size<=inp->input_size;i+=stride)
        {
            for(ll j =0;j+ker_size<=inp->input_size;j+=stride)
            {
                
                rep(l,n_ker)
                {
                    out->input[l][i/stride][j/stride] = 0.0;
                    rep(k,ker_size)
                    {
                        rep(q,ker_size)
                        {
                            out->input[l][i/stride][j/stride] += inp->input[0][i + k][j + q] * kernel[l][k][q];
                        }
                    }
                    out->input[l][i/stride][j/stride] = sigmoid(out->input[l][i/stride][j/stride]);
                }
            }
        }
        return out;
    }
    
    
    void updateWt(lldd &learn_rate)
    {
        if(del_wt==nullptr)
        {
            del_wt = new mat(ker_size,1);
        }
        rep(l,n_ker)
        {
            rep(k,ker_size)
            {
                rep(q,ker_size)
                {
                    del_wt->input[0][k][q] = 0.0;
                }
            }
            for(ll i =0;i+ker_size<=out->input_size;i++)
            {
                for(ll j =0;j+ker_size<=out->input_size;j++)
                {
                        rep(k,ker_size)
                        {
                            rep(q,ker_size)
                            {
                                del_wt->input[0][k][q] += err_rate->input[l][i + k][j+ q]
                                *learn_rate
                                *in->input[0][i*stride + k][j*stride + q]
                                * out->input[l][i + k][j+q];
                            }
                        }
                }
            }
//                        cout << "\nWeight " << l << "\n";
            rep(k,ker_size)
            {
                rep(q,ker_size)
                {
//                    cout <<del_wt->input[0][k][q] << " ";
                    kernel[l][k][q] -= del_wt->input[0][k][q];
                }
//                cout << "\n";
            }

        }
    }
    mat* getErrRate()
    {
        if(next_err_rate==nullptr)
        {
            next_err_rate = new mat(in->input_size,input_dim);
        }
        rep(j,next_err_rate->input_size)
        {
            rep(k,next_err_rate->input_size)
            {
                next_err_rate->input[0][j][k]  = 0.0;
            }
        }
        rep(i,next_err_rate->input_size)
        {
            rep(j,next_err_rate->input_size)
            {
                next_err_rate->input[0][i][j] = 0.0;
                rep(z,n_ker)
                {
                    for(ll k = 0;i+ker_size<in->input_size&&(k<ker_size)&&i+k<err_rate->input_size;k+=stride)
                    {
                        for(ll l = 0;j+ker_size<in->input_size&&(l<ker_size)&&l+j<err_rate->input_size;l+=stride)
                        {
                            next_err_rate->input[0][i][j] += err_rate->input[z][(i+k)/stride ][(j+l)/stride ]
                            *out->input[z][(i+k)/stride ][(j+l)/stride]
                            *kernel[z][ker_size-1-k][ker_size-1-l];
                        }
                    }
                }
            }
        }
        for(ll i =1;i<next_err_rate->n_dim;i++)
        {
            rep(j,next_err_rate->input_size)
            {
                rep(k,next_err_rate->input_size)
                {
                    next_err_rate->input[i][j][k]  = next_err_rate->input[0][j][k];
                }
            }
        }
        return next_err_rate;
    }
    
    void set_err_rate(mat* err)
    {
        if(err_rate==nullptr)
        {
            err_rate = new mat(out->input_size,out->n_dim);
        }
        rep(i,err_rate->n_dim)
        {
            rep(j,err_rate->input_size)
            {
                rep(k,err_rate->input_size)
                {
                    err_rate->input[i][j][k] = err->input[i][j][k];
                }
            }
        }
    }
    
    //BackProp
    void update_kernel_weights(lldd &learn_rate)
    {
        if(del_wt==nullptr)
        {
            del_wt = new mat(ker_size,1);
        }
        rep(l,n_ker)
        {
            rep(k,ker_size)
            {
                rep(q,ker_size)
                {
                    del_wt->input[0][k][q] = 0.0;
                }
            }
            for(ll i =0;i+ker_size<=in->input_size;i+=stride)
            {
                for(ll j =0;j+ker_size<=in->input_size;j+=stride)
                {
                        rep(k,ker_size)
                        {
                            rep(q,ker_size)
                            {
                                del_wt->input[0][k][q] += in->input[0][i + k][j+ q] * learn_rate*err_rate->input[l][i/stride][j/stride];
                            }
                        }
                    
                    // Conv(6) Conv(10) Flat
                    //\// Check this out if we have to have this or not
//                    output[l][i/stride][j/stride] = sigmoid(output[l][i/stride][j/stride]);
                }
            }
//            cout << "\nWeight " << l << "\n";
            rep(k,ker_size)
            {
                rep(q,ker_size)
                {
//                    cout <<del_wt->input[0][k][q] << " ";
                    kernel[l][k][q] -= del_wt->input[0][k][q];
                }
//                cout << "\n";
            }
        }
    }
    

    // ll get_n_ker(){
    //     return n_ker;
    // }
    // bool get_pad(){
    //     return pad;
    // }
    // ll get_pad_len(){
    //     return padding;
    // }
    mat *get_err_rate()
    {
        return err_rate;
    }
    ll get_ker_size()
    {
        return ker_size;
    }
    ll get_stride(){
        return stride;
    }
};

//Pooling Layer
class pool{
    ll stride;
    mat *err_rt;
    ll type;
    ll dim;
    ll output_dim;
    mat *err_rate;
    mat *in;
    mat *out;
    lldd **output;
    public:
//    ~pool()
//    {
//        try {
//            delete(in);
//        } catch (...) {
//            // Do Nothing
//        }
//        try {
//            delete(out);
//        } catch (...) {
//            // Do Nothing
//        }
//    }
    pool()
    {
        printf("Enter dimension of pooling\n");
        scanf(" %lld",&dim);
        printf("Enter stride of pooling\n");
        scanf(" %lld",&stride);
        printf("Enter type of pooling(1->sum, 2->avg, default is max\n");
        scanf(" %lld",&type);
        in = nullptr;
        out = nullptr;
        err_rt = nullptr;
//        cout << in << " in "  << endl;0
    }
    
    mat *scaleup_err(mat *err)
    {
        if(err_rt==nullptr){
            err_rt = new mat(in->input_size,in->n_dim);
        }
        rep(i,in->n_dim)
        {
            rep(j,in->input_size)
            {
                rep(k,in->input_size)
                {
                    err_rt->input[i][j][k] = 0.0;
                }
            }
        }
        rep(l,in->n_dim)
            {
                for(ll i =0;i+dim<=err_rt->input_size;i+=stride)
                {
                    for(ll j =0;j+dim<=err_rt->input_size;j+=stride)
                    {
                        
                        rep(k,dim)
                        {
                            rep(q,dim)
                            {
                                err_rt->input[l][i+k][j+q] += err->input[l][i/stride][j/stride];
                            }
                        }
                    }
                }
            }
//        err->del();
//        err = nullptr;
//        delete err;
        return err_rt;
    }
    
    mat *deflatten(flat* inp)
    {
        if(err_rate){
            err_rate = new mat(output_dim,in->n_dim);
        }
        rep(k,in->n_dim)
        {
            rep(i,output_dim)
            {
                rep(j,output_dim)
                {
                     err_rate->input[k][i][j] = inp->in[k*(output_dim)*(output_dim)+(i*(output_dim))+j];
                    // cout << out[(i*inp->input_size)+j] << "\n";
                    
                }
            }
        }
        return err_rate;
    }
    mat *op(mat *inp)
    {
//        if(out!=nullptr&&*(out->input)!=nullptr)
//        {
//            try {
//                delete(out);
//            } catch (...) {
//                // Do Nothing
//            }
//        }
        if(in==nullptr)
        {
            in = new mat(inp->input_size,inp->n_dim);
        }
        in->copy(inp);
        output_dim = ((inp->input_size - dim)/stride) + 1;
        if(out==nullptr){
//            printf("\nDefining Pool Here\n");
            out = new mat(output_dim,in->n_dim);
//            output= new lldd*[output_dim];
//            rep(i,output_dim)
//            {
//                output[i] = new lldd[output_dim];
//            }
        }
        lldd maxi =LONG_MIN;
        rep(l,in->n_dim)
            {
                for(ll i =0;i+dim<=in->input_size;i+=stride)
                {
                    for(ll j =0;j+dim<=in->input_size;j+=stride)
                    {
                        maxi =LONG_MIN;
                        out->input[l][i/stride][j/stride] = 0.0;
                        rep(k,dim)
                        {
                            rep(q,dim)
                            {
                                out->input[l][i/stride][j/stride] += in->input[l][i/stride + k][j/stride + q];
                                maxi = max(maxi,in->input[l][i/stride + k][j/stride + q]);
                            }
                        }
                        if(type == 2)
                        {
                            out->input[l][i/stride][j/stride] /= (lldd)(dim*dim);
                        }
                        else if(type!=1)
                        {
                            out->input[l][i/stride][j/stride] = maxi;
                        }
                    }
                }
            }
//        if(in!=nullptr)
//        {
//            try {
//                delete(in);
//            } catch (...) {
//                // Do Nothing
//            }
//        }
//        if(out!=nullptr)
//        {
//            out->del();
//            delete out;
//        }
        
        return out;
    }
};


//Single Perceptron
class percept{
    long double *weights;
    long double bias;
    bool init;
    public:
//    ~percept()
//    {
//        try {
//            delete(weights);
//        } catch (...) {
//            // Do Nothing
//        }
//    }
    percept()
    {
        init = 0;
    }
    lldd *get_weight()
    {
        return weights;
    }
    lldd get_bias()
    {
        return bias;
    }
    void set_bias(lldd change)
    {
        bias -= change;
    }
    lldd op(flat *inp)
    {
        if(!init)
        {
            // cout << "Percept Weights\n";
            ll mul = 1;
            if(rand()%2==0)
                {
                    mul = 1;
                }
                else
                {
                    mul = -1;
                }
            bias = gen(engine);
//            bias = ((lldd)mul*(lldd)rand());   // /(lldd)RAND_MAX);
//            bias = 0.0;
            
            weights = new lldd[(inp->in_size)];
            rep(i,inp->in_size)
            {
                if(rand()%2==0)
                {
                    mul = 1;
                }
                else
                {
                    mul = -1;
                }
                weights[i] = weights[i] = (lldd)mul*(lldd)gen(engine);
//                weights[i] = (lldd)mul*(lldd)rand()/5325; //  /(lldd)RAND_MAX;
                 
            }
            init = 1;
        }
//        cout <<"\nBias" <<bias <<"\n";
//        for(ll i =0;i<inp->in_size;i++)
//        {
//            cout << weights[i] << "\n";
//        }
//        cout << " \n\n\n";
        lldd ans = 0;
        rep(i,inp->in_size)
        {
            lldd val = inp->in[i];
            ans+=weights[i]*val;                    //we can add multithreading here to make the code run fast;
        }
        return sigmoid(ans+bias);
    }
};


//Dense Layer
class dense
{
    ll num;
    percept *neu;
    flat *in;
    flat *output;
    flat *error_rate;
    lldd *out;
    public:
    dense()
    {
        printf("Enter Number of Perceptrons in Layer\n");
        scanf(" %lld",&num);
        neu = new percept[num];
        error_rate = new flat(num);
        in = nullptr;
        output = nullptr;
        out = nullptr;
    }
    flat *get_error_rate()
    {
        return error_rate;
    }
//    void set_error_rate(lldd *error)
//    {
//        //NEED TO UPDATE THIS CODE FOR LESS MEMORY
////        if(error_rate!=nullptr)
////        {
////            try{
////                delete error_rate;
////            }
////            catch(...)
////            {
////                printf("Error rate memory error\n");
////            }
////
////        }
//        error_rate = error;
//    }
    void save(ofstream&outdata)
    {
        outdata << num << "\n";
        rep(i,in->in_size)
        {
            rep(j,num)
            {
                if(j!=num-1)
                {
                    outdata << neu[j].get_weight()[i] << ",";
                }
                else
                {
                    outdata << neu[j].get_weight()[i] << "\n";
                }
            }
        }
    }
    ll get_num()
    {
        return num;
    }
    flat *get_out()
    {
        return output;
    }
    percept *get_percept()
    {
        return neu;
    }
    flat *op(flat *inp)
    {
//         if(output!=nullptr)
//         {
//             try {
//                 delete output ;
//             } catch (...) {
//                 // Do Nothing
//                 printf("Exception caught\n");
//             }
//         }
//                if(in!=nullptr)
//                {
//                    try {
//                        delete in;
//                    } catch (...) {
//                        printf("Exception caught\n");
//                    }
//                }
//        cout << "\nInput Entry\n";
//        for(ll i =0;i<inp->in_size;i++)
//        {
//            cout << inp->in[i] << "\n";
//        }
//        cout << "\n\n";
        if(in==nullptr)
        {
            in = new flat(inp->in_size);
//            out = new lldd[num];
        }
        in->copy(inp);
        if(output==nullptr)
        {
            output = new flat(num);
        }
        for(ll i =0;i<num;i++)
        {
            output->in[i] =  neu[i].op(inp);
            // cout << out[i] << "\n";
        }
        
//        try
//        {
//            if(out!=nullptr){
//                delete[] out;
//            }
//        }
//        catch(...){
//
//        }
        // out = nullptr;
//        cout << "\nOutput Entry\n";
//        for(ll i =0;i<output->in_size;i++)
//        {
//            cout << output->in[i] << "\n";
//        }
//        cout << "\n\n";
        return output;
        
    }
    void update_weight(lldd &learn_rate)
    {
        for(ll i =0;i<num;i++)
        {
            neu[i].set_bias(error_rate->in[i]*learn_rate);
            lldd *wth =neu[i].get_weight();
            for(ll j = 0;j<in->in_size;j++)
            {
                wth[j] -= error_rate->in[i]*learn_rate * in->in[j];
//                cout <<"\n ith "<<i << " jth "<<j <<" "<<  error_rate->in[i]*learn_rate * in->in[j]   << "\n";
            }
        }
    }
};

//void work(conv &con,pool &p,dense&d){
//    double **ker;
//    ll ker_siz = 0;
//    ll mul =0;
//    printf("Enter dimension of input\n");
//    scanf("%d",&ker_siz);
//    ker  = new double*[ker_siz];
//        for(ll i =0;i<ker_siz;i++)
//        {
//            ker[i] = new double[ker_siz];
//        }
//        // cout << "Success 1\n";
//        srand(time(0));
//        for(ll i =0;i<ker_siz;i++)
//        {
//            for(ll j=0;j<ker_siz;j++)
//            {
//                if(rand()%2==0)
//                {
//                    mul = 1;
//                }
//                else
//                {
//                    mul = -1;
//                }
//                ker[i][j] = (double)mul*(double)rand()/(double)RAND_MAX;
//                // cout << ker[i][j] << " ";
//            }
//            // cout << "\n";
//        }
//        // cout << "\n\n\n";
//    mat *pt = new mat(ker,ker_siz);
//
//    mat *pt2 = con.conv_op(pt);
//    // for(ll i =0;i<pt2->input_size;i++)
//    // {
//    //     for(ll j =0;j<pt2->input_size;j++)
//    //     {
//    //         cout << pt2->input[i][j] << " ";
//    //     }
//    //     cout << "\n";
//    // }
//
//    pt2 = p.op(pt2);
//    // for(ll i =0;i<pt2->input_size;i++)
//    // {
//    //     for(ll j =0;j<pt2->input_size;j++)
//    //     {
//    //         cout << pt2->input[i][j] << " ";
//    //     }
//    //     cout << "\n";
//    // }
//    flat *pt3 = flatten(pt2);
//    *ker = nullptr;
//    ker = nullptr;
//    pt3 = d.op(pt3);
//    softmax(pt3);
//    for(ll i =0;i<pt3->in_size;i++)
//    {
//            cout << pt3->in[i] << " ";
//    }
//    // pt3 = nullptr;
//    // pt2 = nullptr;
//    // pt = nullptr;
//}

union layers{
    conv *con;
    pool *p;
    dense *d;
};

struct mylayer{
    layers lay;
    ll type;
    mylayer *next;
    mylayer *prev;
    mylayer()
    {
        next = nullptr;
        prev = nullptr;
    }
};


//Super Class
class Layer
{
    mylayer *final_layer_start;
    mylayer *final_layer_end;
    mat *input;
    mat *input_op;
    flat *dense_input;
    flat *output_dense;
    flat *error_rate;
    flat *last_error_rate;
    mat *err_rate;
    mat *last_err;
    lldd learn_rate;
    ll out_cls;
    mylayer *head;
    mylayer *tail;
//    lldd **img_out;
    ll img_dim;
    public:
    ofstream outdata;
    ofstream output_txt;
    ofstream er_rate;
//    ~Layer()
//    {
//        try {
//            delete(input);
//        } catch (...) {
//            // Do Nothing
//        }
//        try {
//            delete(dense_input);
//        } catch (...) {
//            // Do Nothing
//        }
//        try {
//            delete(final_layer_end);
//        } catch (...) {
//            // Do Nothing
//        }
//        try {
//            delete(final_layer_start);
//        } catch (...) {
//            // Do Nothing
//        }
//    }
    Layer()
    {
        outdata.open("saveddata.csv");
        output_txt.open("prediction.csv");
        er_rate.open("er_rate.csv");
        input = nullptr;
        dense_input = nullptr;
        printf("\nPlease Enter learning rate for model\n");
        scanf(" %Lf",&learn_rate);
        img_dim = 28;
        input = new mat(img_dim,1);
        input_op = new mat(img_dim,1);
        last_err = nullptr;
        last_error_rate = nullptr;
        error_rate =nullptr;
        err_rate = nullptr;
    }
    ~Layer()
    {
        output_txt.close();
        er_rate.close();
    }
    void save()
    {
        outdata<<"learn_rate,"<<learn_rate << "\n";
        outdata.close();
    }
    void saveLayer()
    {
        int layy = 0;
        head = final_layer_start;
        while(head!=nullptr)
        {
            if(head->type==1)
            {
                outdata.open("conv"+ to_string(layy) + ".csv",ios_base::ate);
                outdata<<"conv"+ to_string(layy) +",";
                head->lay.con->save(outdata);
                outdata.close();
            }
            else if(head->type==3)
            {
                outdata.open("dense"+ to_string(layy) + ".csv",ios_base::ate);
                outdata<<"dense"+ to_string(layy) +",";
                head->lay.d->save(outdata);
                outdata.close();
            }
            head = head->next;
            layy++;
        }
    }
    void create_layer()
    {
        final_layer_start = nullptr;
        mylayer *ne,*some;
        ne = new mylayer;
        char f;
        do
        {
            printf("Which type of layer do you want? (1 - Kernel, 2 - Pool, 3 - Dense)\n");
            ll tp;
            scanf(" %lld",&tp);
            if(final_layer_start==nullptr)
            {
                final_layer_start = ne;
            }
            else if(tp==1||tp==2||tp==3)
            {
                ne->next = new mylayer;
                some = ne;
                ne = ne->next;
                ne->prev = some;
            }
            if(tp == 1)
            {
                printf("\nPlease Sepcify Layer Properties\n");
                ne->lay.con = new conv;
                ne->type = tp;
            }
            else if(tp==2)
            {
                printf("\nPlease Sepcify Layer Properties\n");
                ne->lay.p = new pool;
                ne->type = tp;
            }
            else if(tp==3)
            {
                printf("\nPlease Sepcify Layer Properties\n");
                ne->lay.d = new dense;
                ne->type = tp;
            }
            else
            {
                printf("\nWrong Value Selected Try Again");
            }
            printf("\nDo you want to continue adding layer (n - to exit)\n");
            scanf(" %c",&f);
            if(f=='n')
            {
                final_layer_end = ne;
            }
        }
        while(f!='n');
    }
    
    void operartion()
    {
        head = final_layer_start;
        input_op = input;
        while(head!=nullptr)
        {
            if(head->type==1&&input!=nullptr)
            {
                input_op = head->lay.con->conv_op(input_op);
            }
            else if(head->type==2&&input!=nullptr)
            {
                input_op=head->lay.p->op(input_op);
            }
            else
            {
                if(dense_input  != nullptr)
                {
                    dense_input = head->lay.d->op(dense_input);
                    
//                    cout << "\nFlattened Output\n";
//                    for(ll i =0;i<dense_input->in_size;i++)
//                    {
//                        cout << dense_input->in[i]<<"\n";
//                    }
//                    cout << "\n\n\n";
                }
                else
                {
                    dense_input = flatten(input_op);
                    dense_input = head->lay.d->op(dense_input);
                }
                
            }
            head = head->next;
        }
        if(dense_input!=nullptr){
//            cout << "Dense OP -> \n";
//            for(ll k=0;k<dense_input->in_size;k++)
//            {
//                cout << dense_input->in[k] << "\n";
//            }
//            output_dense = softmax(dense_input);
//            cout << "\n\nSoftMax\n\n";
//            for(ll i =0;i<output_dense->in_size;i++)
//            {
//                cout << output_dense->in[i] << "\n\n";
//            }
            output_dense = dense_input;
        }
        dense_input = nullptr;
    }
    
    void train(ll dnum)
    {
        tail = final_layer_end;
        last_error_rate = nullptr;
        last_err = nullptr;
//        ll cls;
//        printf("Please mention the class of the image: ");
//        scanf(" %d",&cls);
        
        ll curr = 0;
        while(tail!=nullptr)
        {
            if(tail->type==1&&input!=nullptr)
            {
//                input = tail->lay.con->conv_op(input);
//                Calculate Conv Error
                if(last_err==nullptr)
                {
                    last_err = tail->lay.con->deflatten(last_error_rate);
                }
                tail->lay.con->set_err_rate(last_err);
                tail->lay.con->updateWt(learn_rate);
                last_err = tail->lay.con->getErrRate();
            }
            else if(tail->type==2&&input!=nullptr)
            {
                if(last_err==nullptr)
                {
                    last_err = tail->lay.p->deflatten(last_error_rate);
                }
                last_err = tail->lay.p->scaleup_err(last_err);
            }
            else
            {
                error_rate = tail->lay.d->get_error_rate();
                if(tail==final_layer_end)
                {
                    flat *mytail =tail->lay.d->get_out();
                    lldd error_out_cout = 0.0;
                    for(ll i =0;i<error_rate->in_size;i++)
                    {
                        if(out_cls!=i){
                            error_rate->in[i] = mytail->in[i] - 0.05;
                            
                        }
                        else
                        {
                            error_rate->in[i] = mytail->in[i] - 0.85;
                        }
                            error_out_cout+= abs(error_rate->in[i]);
//                        cout << error_rate->in[i] << " " << i<< " Error Rate for initial layer\n";
                    }
                    if(dnum!=0)
                    er_rate << error_out_cout <<",";
                    else
                    {
                        er_rate << error_out_cout <<"\n";
                    }
//                    cout << " " <<error_out_cout << " ";
                    mytail = nullptr;
                    delete mytail;
                    last_error_rate = tail->lay.d->get_error_rate();
//                    tail->lay.d->set_error_rate(error_rate);
                }
                else
                {
//                    cout << "\nError Rate for layer\n";
//                    error_rate = tail->lay.d->get_error_rate();
//                    lldd *error =tail->next->lay.d->get_error_rate();
                    flat* xx = tail->lay.d->get_out();
                    percept *pp  =tail->next->lay.d->get_percept();
                    for(ll i =0;i<error_rate->in_size;i++)
                    {
                        error_rate->in[i] = 0.0;
                    }
                    for(ll j = 0;j<last_error_rate->in_size;j++)
                    {
                        lldd *wth = pp[j].get_weight();
                        for(ll i =0;i<error_rate->in_size;i++)
                        {
//                            lldd out =
//                            lldd wt = pp[j].get_weight()[i];
                            error_rate->in[i] += last_error_rate->in[j] * wth[i];
                            
                            // (xx->in[j] * (1-(xx->in[j])))
//                            cout << out << " sigmoid for layer "<< tail->next->lay.d->get_error_rate()[j] <<  " " << i+1 <<" \n";
                        }
//                        cout << error_rate[i] << " for layer "<<  curr <<  " weight number -> " << i <<" \n";
                    }
                    xx = nullptr;
                    delete xx;
                    last_error_rate = tail->lay.d->get_error_rate();
//                    tail->lay.d->set_error_rate(error_rate);
                }
                 error_rate = nullptr;
//                dense_input = tail->lay.d->op(dense_input);
            }
            curr++;
            tail=tail->prev;
        }
        tail = final_layer_end;
        while(tail!=nullptr)
        {
            if(tail->type==1&&input!=nullptr)
            {
                //tail->lay.con->update_kernel_weights(learn_rate);
            }
            else if(tail->type==2&&input!=nullptr)
            {
//                input = tail->lay.p->op(input);
            }
            else
            {
                tail->lay.d->update_weight(learn_rate);
            }
            tail=tail->prev;
        }
    }
    
    void load_and_train(ll dnum)
    {
        ifstream fin;
        fin.open("train.csv", ios::in);
        string line, word, temp;
//        mat *fn =
        while(dnum--)
        {
            getline(fin,line);
            stringstream ss(line);
            getline(ss,word,',');
            out_cls = stoi(word);
            ll i = 0;
//            if(input!=nullptr)
//            {
//                input->del();
//                delete input;
//            }
//            input = new mat(img_dim,1);
            while(getline(ss,word,','))
            {
                input->input[0][i/img_dim][i%img_dim] = stoi(word);
                input->input[0][i/img_dim][i%img_dim] = (lldd)input->input[0][i/img_dim][i%img_dim]/(lldd)255.0;
                i++;
            }
//            fn->make(img_out,img_dim);
            
//            rep(i,img_dim){
//                rep(j,img_dim){
//                    input[i][j]=img_out[i][j];
//                }
//            }
//            input = new mat(ker,28);
            operartion();
            ll ans = -1;
            long double maxi =0;
            for(ll k=0;k<output_dense->in_size;k++)
            {
                if(maxi<output_dense->in[k]){
                    maxi = output_dense->in[k];
                    ans = k;
                }
            }
//            cout << "\n\nSoftMax\n\n";
//            for(ll i =0;i<output_dense->in_size;i++)
//            {
//                cout << output_dense->in[i] << "\n";
//            }
//            cout << ans  << " " << out_cls <<"\n";
                train(dnum);
            
        }
//        er_rate << "\n";
//        try
//        {
//            fn->del();
//            delete fn;
//        }
//        catch(...)
//        {
//            //
//        }
        fin.close();
    }
    void test(ll dnum)
    {
        long double ddnum = dnum;
        long correct = 0;
        ifstream fin;
        fin.open("train.csv", ios::in);
        string line, word, temp;
//        mat *ff = new mat(img_out,img_dim);
        while(dnum--)
        {
            getline(fin,line);
            stringstream ss(line);
            getline(ss,word,',');
            out_cls = stoi(word);
            ll i = 0;
//            if(input!=nullptr)
//            {
//                input->del();
//                delete input;
//            }
//            input = new mat(img_dim,1);
            while(getline(ss,word,','))
            {
                input->input[0][i/img_dim][i%img_dim] = (lldd)stoi(word)/(lldd)255.0;
                i++;
            }
//            ff->make(img_out,img_dim);
//            if(input!=nullptr)
//            {
//                input->del();
//                delete input;
//            }
//            input = new mat(img_out,img_dim);
//            output_dense = nullptr;
//            dense_input = nullptr;
            operartion();
            ll ans = -1;
            long double maxi =0;
//            cout << "\n\nSoftMax\n\n";
            for(ll i =0;i<output_dense->in_size;i++)
            {
//                cout << output_dense->in[i] << "\n";
                if(maxi<=output_dense->in[i])
                {
                    maxi = output_dense->in[i];
                    ans = i;
                }
            }
            output_txt << ans  << "," << out_cls <<"\n";
//            cout << ans  << " " << out_cls <<"\n";
            if(ans==out_cls)
            {
                correct++;
            }
        }
//        output_txt <<"Error Rate :" <<","<< (1.0 - (correct/ddnum))*100.0 << "\n";
        cout << "Error Rate : " << (1.0 - (correct/ddnum))*100.0 << "\n";
//        try
//        {
//            ff->del();
//            delete ff;
//        }
//        catch(...)
//        {
//            //
//        }
        fin.close();
    }
    
};


int main()
{
    // conv con;
    // pool p;
    // dense d;
    // char f;
    // do
    // {
    //     work(con,p,d);
    //     printf("Do you want to continue\n");
    //     scanf(" %c",&f);
    // }
    // while(f!='n');
//    read();
    Layer r;
    r.create_layer();
//    char f;
    ll iter = 0, num = 0;
    printf("Enter number of training iterations\n");
    scanf(" %lld",&iter);
    printf("\nEnter test data size\n");
    scanf(" %lld",&num);
    r.outdata <<"Max Iteration," <<iter << ",Test Data Size," << num <<",";
    r.save();
    do
    {
        r.load_and_train(num);
        cout << "Current Itr -> " << iter << "\n";
    }
    while(iter--);
    r.saveLayer();
    r.test(40000);
//    r.test();
//
    
    
    
//    do
//    {
//        r.create_input();
//        r.operartion();
//        r.train();
//        printf("\nDo you want to continue with new input for training\n");
//        scanf(" %c",&f);
//    }
//    while(f!='n');
//
    return 0;
}
